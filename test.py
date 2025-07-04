import os
import torch
import argparse
import logging
import json  #
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize
from models.causal_vqa_model import CausalVQAModel
from models.m3ae import M3AE
from utils.dataloader import VQADataLoader
from train import compute_score_with_logits, prepare_batch_data
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Medical VQA System Test')

    # Data-related parameters
    parser.add_argument('--data_dir', type=str, default='data/slake', help='Root data directory')
    parser.add_argument('--image_dir', type=str, default='data/slake/imgs', help='Image directory')
    parser.add_argument('--test_json', type=str, default='data/slake/test.json', help='Test data JSON')
  
    parser.add_argument('--load_path', type=str, default='pretrained_weights/m3ae.ckpt', help='Pretrained weights path')
    parser.add_argument('--ae_weight_path', type=str, default='pretrained_weights/pretrained_ae.pth',
                        help='Autoencoder weights path')
    parser.add_argument('--maml_weight_path', type=str, default='pretrained_weights/pretrained_maml.weights',
                        help='MAML weights path')
    parser.add_argument('--embeddings_dir', type=str, default='data/slake/embeddings_all', help='Embeddings directory')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, default='', help='Model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    parser.add_argument('--vocab', type=str, default='roberta', help='Vocabulary')
    parser.add_argument('--image_size', type=int, default=384, help='Image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--max_length', type=int, default=32, help='Max sequence length')
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden dimension')
    parser.add_argument('--num_top_layer', type=int, default=6, help='attention layer')
    parser.add_argument('--input_image_embed_size', type=int, default=768, help='Visual feature dimension')
    parser.add_argument('--input_text_embed_size', type=int, default=768, help='Question feature dimension')

    parser.add_argument('--seed', type=int, default=105, help='Random seed')

    return parser.parse_args()


def test_accuracy(args):
    """Evaluate model accuracy on the test set"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Configure data loader
    data_config = {
        'data_dir': args.data_dir,
        'image_dir': args.image_dir,
        'embeddings_dir': args.embeddings_dir,
        'test_json': args.test_json,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'tokenizer': args.vocab,
        'image_size': args.image_size,
        'max_length': args.max_length,
        'device': str(device)
    }

    # Initialize data loader
    logger.info("Initializing data loader...")
    data_loader = VQADataLoader(data_config)
    loaders = data_loader.get_loaders()
    test_loader = loaders.get('test')

    if test_loader is None or len(test_loader) == 0:
        logger.error("Test data loading failed!")
        return

    # Initialize model
    logger.info("Initializing model...")
    model_config = {
        'hidden_size': args.hidden_size,
        'num_hid': data_loader.get_answer_vocab()['vocab_size'],
        'input_image_embed_size': args.input_image_embed_size,
        'input_text_embed_size': args.input_text_embed_size,
        'num_top_layer': args.num_top_layer,
        'visual_backbone': 'ViT-B/16',
        'image_size': args.image_size,
        'patch_size': args.patch_size,
        'load_path': args.load_path,
        'ae_weight_path': args.ae_weight_path,
        'maml_weight_path': args.maml_weight_path,
    }
    model = CausalVQAModel(model_config)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    logger.info("Start evaluation...")
    total_correct = 0
    total_samples = 0
    criterion = F.binary_cross_entropy_with_logits
    
    # Initialize prediction results list
    predictions_list = []
    
    # Initialize inference performance statistics
    total_inference_time = 0.0
    total_samples_processed = 0
    batch_times = []
    sample_times = []
    
    with torch.no_grad():
        # Initialize counters
        total_correct = 0
        total_loss = 0
        total_samples = 0
        closed_correct = 0  # Number of correct Yes/No questions
        closed_total = 0    # Total number of Yes/No questions
        open_correct = 0    # Number of correct open-ended questions
        open_total = 0      # Total number of open-ended questions
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Record batch start time
            batch_start_time = time.time()
            
            images = batch['images'].to(device)
            questions = batch['questions']['input_ids'].to(device)
            attention_mask = batch['questions']['attention_mask'].to(device)
            do_questions = batch['do_questions']['input_ids'].to(device)
            do_attention_mask = batch['do_questions']['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            pattern_embedding = batch.get('pattern_embedding', None)
            entity_embedding = batch.get('entity_embedding', None)
            ae_images = batch.get('ae_images', None)
            maml_images = batch.get('maml_images', None)

            if pattern_embedding is not None:
                pattern_embedding = pattern_embedding.to(device)
            if entity_embedding is not None:
                entity_embedding = entity_embedding.to(device)
            if ae_images is not None:
                ae_images = ae_images.to(device)
            if maml_images is not None:
                maml_images = maml_images.to(device)

            answer_types = batch['answer_types']
            question_texts = batch['question_texts']
            answer_texts = batch['answer_texts']
           
            # Record inference start time
            inference_start_time = time.time()
            
            # Forward pass
            logits = model(
                images,
                questions,
                attention_mask,
                do_questions,
                do_attention_mask,
                ae_images=ae_images,
                maml_images=maml_images,
                pattern_embedding=pattern_embedding,
                entity_embedding=entity_embedding,
                training=False
            )
       
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            
            # Record batch end time
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            
            # Inference performance statistics
            batch_size = images.size(0)
            total_inference_time += inference_time
            total_samples_processed += batch_size
            batch_times.append(batch_time)
            sample_times.extend([inference_time / batch_size] * batch_size)  # Average inference time per sample
            
            loss = criterion(logits, targets)
            total_loss += loss.item()
            # Calculate prediction results
            pred_indices = torch.max(logits, 1)[1].data
            batch_scores = compute_score_with_logits(logits, targets)
            
            # Calculate accuracy by answer type
            for i, (pred, score, ans_type) in enumerate(zip(pred_indices, batch_scores, answer_types)):
                total_samples += 1
                total_correct += score.sum().item()
                
                # Determine if it is a closed (Yes/No) question
                ans_lower = ans_type.lower()
                if ans_lower == 'closed':
                    closed_total += 1
                    closed_correct += score.sum().item()
                else:
                    open_total += 1
                    open_correct += score.sum().item()
            
            # Print performance statistics every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_batch_time = np.mean(batch_times[-10:])
                avg_inference_time = np.mean(sample_times[-10*batch_size:])
                throughput = batch_size / avg_batch_time
                logger.info(f"Batch {batch_idx + 1}: Avg batch time {avg_batch_time:.4f}s, "
                          f"Avg inference time {avg_inference_time:.4f}s, Throughput {throughput:.2f} samples/s")

    # Calculate various accuracies
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    closed_accuracy = closed_correct / closed_total if closed_total > 0 else 0
    open_accuracy = open_correct / open_total if open_total > 0 else 0
    
    # Calculate inference performance statistics
    avg_inference_time_per_sample = total_inference_time / total_samples_processed if total_samples_processed > 0 else 0
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    throughput_samples_per_sec = total_samples_processed / total_inference_time if total_inference_time > 0 else 0
    throughput_batches_per_sec = len(batch_times) / sum(batch_times) if batch_times else 0
    
    # Calculate latency statistics
    p50_inference_time = np.percentile(sample_times, 50) if sample_times else 0
    p90_inference_time = np.percentile(sample_times, 90) if sample_times else 0
    p95_inference_time = np.percentile(sample_times, 95) if sample_times else 0
    p99_inference_time = np.percentile(sample_times, 99) if sample_times else 0
    
    # Print results
    logger.info(f"Overall test accuracy: {overall_accuracy:.2%}")
    logger.info(f"Closed (Yes/No) question accuracy: {closed_accuracy:.2%} ({closed_correct}/{closed_total})")
    logger.info(f"Open-ended question accuracy: {open_accuracy:.2%} ({open_correct}/{open_total})")
    logger.info(f"Average loss: {total_loss / total_samples:.4f}")
    
    # Print inference performance results
    logger.info("=" * 50)
    logger.info("Inference performance statistics:")
    logger.info(f"Total inference time: {total_inference_time:.4f}s")
    logger.info(f"Total samples: {total_samples_processed}")
    logger.info(f"Average inference time/sample: {avg_inference_time_per_sample:.4f}s")
    logger.info(f"Average batch time: {avg_batch_time:.4f}s")
    logger.info(f"Throughput: {throughput_samples_per_sec:.2f} samples/s")
    logger.info(f"Batch throughput: {throughput_batches_per_sec:.2f} batches/s")
    logger.info("Latency statistics:")
    logger.info(f"  P50: {p50_inference_time:.4f}s")
    logger.info(f"  P90: {p90_inference_time:.4f}s")
    logger.info(f"  P95: {p95_inference_time:.4f}s")
    logger.info(f"  P99: {p99_inference_time:.4f}s")
    logger.info("=" * 50)

    return overall_accuracy


if __name__ == "__main__":
    args = parse_args()
    test_accuracy(args)