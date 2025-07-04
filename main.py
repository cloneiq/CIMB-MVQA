import os
import json
import torch
import itertools
import numpy as np
import argparse
import logging
import torch.nn.functional as F
import torchvision
import torch.nn as nn

torchvision.disable_beta_transforms_warning()
from tqdm import tqdm
import pickle

from train import train_epoch, validate
from utils.dataloader import VQADataLoader
from utils.pseudo_mask import PseudoOrganMaskGenerator
from models.causal_vqa_model import CausalVQAModel
from models.m3ae import M3AE
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from transformers import get_cosine_schedule_with_warmup
import math
from torch.optim.lr_scheduler import LambdaLR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VQADataLoader')

# Environment variable settings
for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if env_var in os.environ:
        os.environ.pop(env_var)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable blocking for CUDA operation

def parse_args():
    parser = argparse.ArgumentParser(description='training')

    # Data-related parameters
    parser.add_argument('--data_dir', type=str, default='data/rad', help='Root directory of data')
    parser.add_argument('--image_dir', type=str, default='data/rad/imgs', help='Image directory')
    parser.add_argument('--train_json', type=str, default='data/rad/train.json', help='Training data JSON')
    parser.add_argument('--val_json', type=str, default='data/slake/test.json', help='Validation data JSON')
    parser.add_argument('--test_json', type=str, default='data/slake/test.json', help='Test data JSON')

    # Model-related parameters
    parser.add_argument('--vocab', type=str, default='roberta', help='Vocabulary')
    parser.add_argument('--image_size', type=int, default=384, help='Image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--max_length', type=int, default=32, help='Maximum sequence length')
   
    parser.add_argument('--hidden_dim', type=int, default=768, help='hidden dimension')
    parser.add_argument('--num_top_layer', type=int, default=6, help='attention layer')
    parser.add_argument('--input_image_embed_size', type=int, default=768, help='Visual feature dimension')
    parser.add_argument('--input_text_embed_size', type=int, default=768, help='Question feature dimension')
    parser.add_argument('--finetune', type=bool, default=False, help='Whether to finetune the pretrained model')
    parser.add_argument('--ae_weight_path', type=str, default='pretrained_weights/pretrained_ae.pth',
                        help='Autoencoder weights path')
    parser.add_argument('--maml_weight_path', type=str, default='pretrained_weights/pretrained_maml.weights',
                        help='MAML weights path')
    parser.add_argument('--load_path', type=str, default='pretrained_weights/m3ae.ckpt',
                        help='Pretrained weights path')
    parser.add_argument('--embeddings_dir', type=str, default='data/slake/embeddings_all', help='Embeddings directory')

    # Training-related parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=24, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=75, help='Number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--warmup_type', type=str, default='sigmoid', help='Warmup type')
    parser.add_argument('--lam_const', type=float, default=1, help='Consistency loss weight')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=3, help='Gradient clipping')
    parser.add_argument('--early_stop', type=int, default=5, help='Early stopping epochs')
    parser.add_argument('--seed', type=int, default=105, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=5, help='Logging interval')
    parser.add_argument('--device', type=str, default='cuda', help='Device (leave blank for auto selection)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'infer'], help='Run mode')
    parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint path (for evaluation or inference)')
    parser.add_argument('--rebuild_vocab', action='store_true', help='Rebuild vocabulary')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Model save directory')
   
    # Visualization parameters
    parser.add_argument('--visualize_every', type=int, default=5, help='Visualize every N epochs')

    return parser.parse_args()


def train(args, device):
    """Execute model training process"""
    import torch.nn as nn
    import torch.optim as optim

    # Create save directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    logger.info("===== Start training preparation =====")

    # Configure dataloader parameters
    data_config = {
        'data_dir': args.data_dir,
        'image_dir': args.image_dir,
        'embeddings_dir': args.embeddings_dir,
        'train_json': args.train_json,
        'val_json': args.val_json,
        'test_json': args.test_json,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'image_size': args.image_size,
        'max_length': args.max_length,
        'tokenizer': args.vocab,
        'rebuild_vocab': args.rebuild_vocab,
        'device': str(device)
    }

    # Initialize dataloader
    logger.info("Initializing dataloader...")
    data_loader = VQADataLoader(data_config)
    loaders = data_loader.get_loaders()
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')

    # Get number of answer classes
    answer_vocab = data_loader.get_answer_vocab()
    num_classes = answer_vocab['vocab_size']
    logger.info(f"Number of answer classes: {num_classes}")

    # Model configuration
    model_config = {
        'input_image_embed_size': args.input_image_embed_size,
        'input_text_embed_size': args.input_text_embed_size,
        'num_top_layer': args.num_top_layer,
        'hidden_size': args.hidden_dim,
        'num_hid': num_classes,
        'dropout': 0.1,
        'visual_backbone': 'ViT-B/16',
        'image_size': args.image_size,
        'ae_weight_path': args.ae_weight_path,
        'maml_weight_path': args.maml_weight_path,
        'load_path': args.load_path,
        'patch_size': args.patch_size
    }

    train_config = {
        "lam_const": args.lam_const,
        "warmup_epochs": args.warmup_epochs,
        "warmup_type": args.warmup_type,
    }

    # Initialize model
    logger.info("Initializing model...")
    model = CausalVQAModel(model_config)
    # model = M3AE(model_config)
    model = model.to(device)
    # Print number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Define loss function and optimizer
    structure_mask_generator = PseudoOrganMaskGenerator(image_size=(args.image_size, args.image_size),
                                                        patch_size=args.patch_size)
    criterion = F.binary_cross_entropy_with_logits

    optimizer = optim.AdamW([
        # Pretrained modules (small learning rate)
        {"params": model.language_encoder.parameters(), "lr": 5e-6},
        {"params": model.vision_encoder.parameters(), "lr": 5e-6},
        {"params": model.auto_encoder.parameters(), "lr": 5e-6},
        {"params": model.maml_model.parameters(), "lr": 5e-6},

        # New modules (larger learning rate)
        {"params": model.multi_modal_language_proj.parameters(), "lr": 5e-6},
        {"params": model.multi_modal_vision_proj.parameters(), "lr": 5e-6},
        {"params": model.multi_modal_do_language_proj.parameters(), "lr": 5e-5},

        {"params": model.modality_type_embeddings.parameters(), "lr": 5e-6},

        {"params": model.multi_modal_language_layers.parameters(), "lr": 5e-6},
        {"params": model.multi_modal_vision_layers.parameters(), "lr": 5e-6},
        {"params": model.multi_modal_do_language_layers.parameters(), "lr": 5e-5},
        {"params": model.multi_modal_vision_post_layers.parameters(), "lr": 5e-6},
        # {"params": model.multi_modal_language_post_layers.parameters(), "lr": 5e-6},

        {"params": model.multi_modal_language_pooler.parameters(), "lr": 5e-6},
        {"params": model.multi_modal_vision_pooler.parameters(), "lr": 5e-6},
        {"params": model.multi_modal_do_language_pooler.parameters(), "lr": 5e-5},
        {"params": model.multi_modal_graph_pooler.parameters(), "lr": 5e-5},

        {"params": model.convert.parameters(), "lr": 5e-5},
        {"params": model.pure_vision_MLP.parameters(), "lr": 5e-5},
        {"params": model.vision_embedding_MLP.parameters(), "lr": 5e-5},

        {"params": model.do_position_embeddings.parameters(), "lr": 5e-5},
        {"params": model.do_token_type_embeddings.parameters(), "lr": 5e-5},
        {"params": [model.visual_modality_embedding], "lr": 5e-5},

        {"params": model.text_modal_graph.parameters(), "lr": 5e-5},
        {"params": model.vqa_head.parameters(), "lr": 5e-5},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-7)

    epoch = 1
    best_val_score = 0
    early_stop_count = 0
    
    # Initialize loss record list
    training_history = []

    # Check if validation set is available
    has_val_data = val_loader is not None and len(val_loader) > 0
    if not has_val_data:
        logger.warning("No validation set provided, will use training loss to save the best model")

    # Training loop
    logger.info("Start training...")
    while epoch < args.epochs:
        # Train for one epoch
        model.train()

        # Train one epoch
        train_loss, train_acc, loss_info = train_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            structure_mask_generator=structure_mask_generator,
            device=device,
            epoch=epoch,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            config=train_config
        )
        
        # Record training loss information
        epoch_record = {
            'epoch': epoch,
            'total_cls_loss': loss_info['total_cls_loss'],
            'total_factor_loss': loss_info['total_factor_loss'],
            'total_loss': loss_info['total_loss'],
            'train_accuracy': loss_info['accuracy'],
            'enable_causal': loss_info['enable_causal'],
            'learning_rate': loss_info['learning_rate']
        }

        run_validation = has_val_data and epoch % args.val_freq == 0
        if run_validation:
            val_loss, val_acc = validate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device
            )
            current_score = val_acc
            
            # Add validation info to record
            epoch_record.update({
                'val_loss': val_loss,
                'val_accuracy': val_acc
            })
            
            logger.info(
                f"Epoch {epoch}: Train loss {train_loss:.4f}, Train accuracy {train_acc:.2f}%, Val loss {val_loss:.4f}, Val accuracy {val_acc:.2f}%")

            # Only save model if validation accuracy improves
            if current_score > best_val_score:
                logger.info(f"Validation performance improved: {best_val_score:.2f}% -> {current_score:.2f}%. Saving model...")
                best_val_score = current_score

                # Only save the best model (overwrite)
                best_model_path = os.path.join(args.save_dir, '11111.pth')
                torch.save(model.state_dict(), best_model_path)

                # Reset early stopping counter
                early_stop_count = 0
            else:
                early_stop_count += 1
                logger.info(f"Validation performance did not improve. Early stop count: {early_stop_count}/{args.early_stop}")
        else:
            # Do not save model and continue training for non-validation epochs
            logger.info(f"Epoch {epoch}: Train loss {train_loss:.4f}, Train accuracy {train_acc:.2f}%")
        
        # Add current epoch record to history
        training_history.append(epoch_record)
        
        # Save training history to file
        history_file = os.path.join(args.save_dir, 'our_training_history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Training history saved to: {history_file}")

        # Check early stopping
        if args.early_stop > 0 and early_stop_count >= args.early_stop:
            logger.info(f"Early stopping triggered, no performance improvement for {args.early_stop} epochs. Stopping training.")
            break

        epoch += 1

    logger.info(f"Training complete! Best performance: {best_val_score:.2f}%")
    
    # Save final training history
    final_history_file = os.path.join(args.save_dir, 'final_our_training_history.json')
    with open(final_history_file, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Final training history saved to: {final_history_file}")
    
    return model


if __name__ == "__main__":
    import random
    args = parse_args()
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)
