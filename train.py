import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import logging
import itertools
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from utils.lossfc_tools import get_current_consistency_weight

logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax to get predicted class indices
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)  # create one-hot encoding
    scores = (one_hots * labels)  # compute label value at predicted class
    return scores


def binary_ce_with_hard_negative(logits, targets, neg_smooth=0.3, neg_weight=0.5):
    batch_size = logits.size(0) // 2
    num_classes = logits.size(1)
    orig_logits = logits[:batch_size]
    neg_logits = logits[batch_size:]
    orig_targets = targets[:batch_size]
    neg_targets = targets[batch_size:].clone()

    orig_loss = F.binary_cross_entropy_with_logits(orig_logits, orig_targets)
    batch_score = compute_score_with_logits(orig_logits, orig_targets)

    # Create smoothed label matrix
    smoothed_targets = torch.zeros_like(neg_targets)
    positive_indices = torch.argmax(neg_targets, dim=1)
    
    # Get original value of positive class for each sample
    original_positive_values = torch.gather(neg_targets, 1, positive_indices.unsqueeze(1)).squeeze(1)
    
    # Compute value assigned to other classes
    other_value = original_positive_values.unsqueeze(1) * neg_smooth / (num_classes - 1)
    
    for i in range(batch_size):
        # Set all classes to small value
        smoothed_targets[i, :] = other_value[i]
        
        # Set positive class to original value times retention ratio
        original_value = original_positive_values[i]
        smoothed_targets[i, positive_indices[i]] = original_value * (1 - neg_smooth)

    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, smoothed_targets)
    total_loss = (1 - neg_weight) * orig_loss + neg_weight * neg_loss

    return total_loss, batch_score


def prepare_batch_data(batch, device, duplicate_text=True, duplicate_mask=True):
    processed_data = {}

    # Process image data - concatenate three types of images
    images = batch['images'].to(device)
    pos_images = batch['pos_images'].to(device)
    neg_images = batch['neg_images'].to(device)
    processed_data['combined_images'] = torch.cat([images, pos_images, neg_images], dim=0)
    processed_data['images'] = images
    processed_data['pos_images'] = pos_images
    processed_data['neg_images'] = neg_images

    # Process text data - duplicate
    questions_ids = batch['questions']['input_ids'].to(device)
    attention_mask = batch['questions']['attention_mask'].to(device)
    do_questions_ids = batch['do_questions']['input_ids'].to(device)
    do_attention_mask = batch['do_questions']['attention_mask'].to(device)

    if duplicate_text:
        processed_data['questions_ids'] = torch.cat([questions_ids, questions_ids], dim=0)
        processed_data['attention_mask'] = torch.cat([attention_mask, attention_mask], dim=0)
        processed_data['do_questions_ids'] = torch.cat([do_questions_ids, do_questions_ids], dim=0)
        processed_data['do_attention_mask'] = torch.cat([do_attention_mask, do_attention_mask], dim=0)
    else:
        processed_data['questions_ids'] = questions_ids
        processed_data['attention_mask'] = attention_mask
        processed_data['do_questions_ids'] = do_questions_ids
        processed_data['do_attention_mask'] = do_attention_mask

    targets = batch['targets'].to(device)
    if duplicate_text:
        processed_data['targets'] = torch.cat([targets, targets], dim=0)
    else:
        processed_data['targets'] = targets

    optional_fields = ['ae_images', 'maml_images', 'pattern_embedding', 'entity_embedding']
    for field in optional_fields:
        if field in batch and batch[field] is not None:
            tensor = batch.get(field).to(device)
            if duplicate_text:
                processed_data[field] = torch.cat([tensor, tensor], dim=0)
            else:
                processed_data[field] = tensor

    if 'mask' in batch and batch['mask'] is not None:
        structure_mask = batch.get('mask').to(device)
        structure_mask = structure_mask.squeeze(1)
        if duplicate_mask:
            processed_data['structure_mask'] = torch.cat([structure_mask, structure_mask], dim=0)
        else:
            processed_data['structure_mask'] = structure_mask

    processed_data['batch_size'] = processed_data['questions_ids'].size(0)

    return processed_data


def train_epoch(model, data_loader, criterion, optimizer, scheduler, structure_mask_generator, device, epoch,
                grad_clip=None, log_interval=10,
                config=None, topv=-1, ture_topk_ratio=0.35):
    model.train()

    total_cls_loss = 0.0
    total_score = 0.0
    # Add necessary initialization
    total_current = 0
    total_factor_loss = 0.0
    start_time = time.time()
    
    # Set which epoch to start mask calculation and causal reasoning
    causal_start_epoch = config.get("causal_start_epoch", 5)
    enable_causal = epoch >= causal_start_epoch
    
    # Set sub-batch size ratio for causal training
    causal_batch_ratio = config.get("causal_batch_ratio", 0.25)  # Default is 1/4 of original batch

    for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} Training")):
        # =========== 1) Prepare input ===========
        images = batch['images'].to(device)
        pos_images = batch['pos_images'].to(device)
        neg_images = batch['neg_images'].to(device)
        combined_images = torch.cat([images, pos_images, neg_images], dim=0)
        data = prepare_batch_data(batch, device, duplicate_text=False)
        questions_ids = data['questions_ids']
        attention_mask = data['attention_mask']
        targets = data['targets']
        do_questions_ids = data['do_questions_ids']
        do_attention_mask = data['do_attention_mask']
        pattern_embedding = data.get('pattern_embedding')
        entity_embedding = data.get('entity_embedding')
        ae_images = data.get('ae_images')
        maml_images = data.get('maml_images')

        batch_size = questions_ids.size(0)

        # batch_size = data['batch_size']
        # =========== 2) Forward ===========
        # Assume model returns (logits_o, logits_a, decoder, ori_vis_features, aug_vis_features)
        logits, cf_loss, v_feats, q_feats, decoder = model(
            combined_images,
            questions_ids,
            attention_mask,
            do_questions_ids,
            do_attention_mask,
            ae_images=ae_images,
            maml_images=maml_images,
            pattern_embedding=pattern_embedding,
            entity_embedding=entity_embedding,
            epoch=epoch,
            causal_start_epoch=causal_start_epoch,
            training=True
        )

        # =========== 3) Main loss (classification) ===========
        loss = criterion(logits, targets)
        
        # AE reconstruction loss (if present)
        if decoder is not None and ae_images is not None:
            ae_criterion = nn.MSELoss()
            loss += 0.0001 * ae_criterion(ae_images, decoder)

        if not enable_causal:
            loss.backward()
            if grad_clip:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            
            # Note: Do not count consistency loss, as it is not used in base training
            total_current += batch_size
            total_score += compute_score_with_logits(logits, targets).sum().item()
            total_cls_loss += loss.item() * batch_size
            
            if (i + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"| Batch {i + 1}/{len(data_loader)} | {elapsed * 1000 / log_interval:.2f} ms/batch | "
                    f"Total Loss {total_cls_loss / total_current:.4f} | "
                    f"Score {total_score / total_current * 100:.2f}% | "
                    f"LR {lr:.6f}| Phase: Base Training"
                )
                start_time = time.time()
            continue
            
        # =========== Causal reasoning part after epoch 5 ===========
        
        # 1. Compute initial feature contribution (for the whole batch)
        visual_grad = torch.autograd.grad((logits * (targets > 0).float()).sum(), v_feats, create_graph=True)[0]
        q_grad = torch.autograd.grad((logits * (targets > 0).float()).sum(), q_feats, create_graph=True)[0]
        
        # =========== Causal independence decomposition - feature consistency loss ===========
        # Apply consistency loss (only in causal training phase, computed once per batch)
        const_weight = get_current_consistency_weight(
            epoch=epoch - causal_start_epoch,
            weight=config.get("lam_const", 1),
            rampup_length=config.get("warmup_epochs", 5),
            rampup_type=config.get("warmup_type", "sigmoid")
        )
        loss += const_weight * cf_loss
        
        # Backpropagate total loss including consistency loss
        loss.backward()
        if grad_clip:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        
        total_factor_loss += const_weight * cf_loss.item() * batch_size
        
        # Compute sub-batch size
        sub_batch_size = max(1, int(batch_size * causal_batch_ratio))
        num_sub_batches = (batch_size + sub_batch_size - 1) // sub_batch_size
        
        # Causal training for each sub-batch
        for sub_idx in range(num_sub_batches):
            start_idx = sub_idx * sub_batch_size
            end_idx = min((sub_idx + 1) * sub_batch_size, batch_size)
            current_sub_batch_size = end_idx - start_idx
            
            # Get sub-batch data (use detach to prevent duplicate gradient computation)
            sub_v_feats = v_feats[start_idx:end_idx].detach()
            sub_q_feats = q_feats[start_idx:end_idx].detach()
            sub_visual_grad = visual_grad[start_idx:end_idx].detach()
            sub_q_grad = q_grad[start_idx:end_idx].detach()
            sub_logits = logits[start_idx:end_idx]
            sub_targets = targets[start_idx:end_idx]
            
            # =========== 5) Causal reasoning ==========
            # Prepare sub-batch data for causal reasoning
            sub_batch = {}
            for key in ['images', 'pos_images', 'neg_images']:
                sub_batch[key] = batch[key][start_idx:end_idx]
            sub_batch['questions'] = {
                'input_ids': batch['questions']['input_ids'][start_idx:end_idx],
                'attention_mask': batch['questions']['attention_mask'][start_idx:end_idx]
            }
            sub_batch['do_questions'] = {
                'input_ids': batch['do_questions']['input_ids'][start_idx:end_idx],
                'attention_mask': batch['do_questions']['attention_mask'][start_idx:end_idx]
            }
            sub_batch['targets'] = batch['targets'][start_idx:end_idx]
            
            # Process optional fields
            optional_fields = ['ae_images', 'maml_images', 'pattern_embedding', 'entity_embedding', 'mask']
            for field in optional_fields:
                if field in batch and batch[field] is not None:
                    sub_batch[field] = batch[field][start_idx:end_idx]

            sub_data = prepare_batch_data(sub_batch, device)
            targets_all = sub_data['targets']
            ae_images = sub_data.get('ae_images')
            maml_images = sub_data.get('maml_images')
            questions_ids = sub_data['questions_ids']
            pattern_embedding = sub_data.get('pattern_embedding')
            entity_embedding = sub_data.get('entity_embedding')

            v_mask_pre = torch.zeros(current_sub_batch_size, sub_v_feats.size(1)).to(sub_v_feats.device)  # [B, 577]
            q_mask_pre = torch.zeros(current_sub_batch_size, sub_q_feats.size(1)).to(sub_q_feats.device)  # [B, 577]
            # Compute importance score for each patch (using previously computed gradients)
            visual_grad_cam = sub_visual_grad.sum(2)  # [B, 577]
            visual_grad_patch = visual_grad_cam[:, 1:]  # Exclude CLS, only take 576 patches
            q_grad_cam = sub_q_grad.sum(2)
            q_grad_patch = q_grad_cam[:, 1:]
            # Sort patch gradient importance
            _, v_grad_indices = visual_grad_patch.sort(1, descending=True)
            v_grad_indices = v_grad_indices + 1  # Restore real index (CLS is 0)

            _, q_grad_indices = q_grad_patch.sort(1, descending=True)
            q_grad_indices = q_grad_indices + 1  # Restore real index (CLS is 0)

            if topv == -1:
                # softmax score + cumulative selection
                v_grad_score = visual_grad_patch.gather(1, v_grad_indices - 1)  # gather needs -1
                q_grad_score = q_grad_patch.gather(1, q_grad_indices - 1)
                v_grad_score = F.softmax(v_grad_score * 10, dim=1)
                q_grad_score = F.softmax(q_grad_score * 10, dim=1)
                v_grad_sum = torch.cumsum(v_grad_score, dim=1)
                v_grad_mask = (v_grad_sum <= 0.85).long()
                v_grad_mask[:, 0] = 1  # at least select one
                q_grad_sum = torch.cumsum(q_grad_score, dim=1)
                q_grad_mask = (q_grad_sum <= 0.85).long()
                q_grad_mask[:, 0] = 1  # at least select one
                for x in range(current_sub_batch_size):
                    num = len(torch.nonzero(v_grad_mask[x]))
                    v_mask_pre[x].scatter_(0, v_grad_indices[x, :num], 1)
                    num = len(torch.nonzero(q_grad_mask[x]))
                    q_mask_pre[x].scatter_(0, q_grad_indices[x, :num], 1)
            else:
                # topv selection
                v_star = v_grad_indices[:, :topv]
                v_mask_pre.scatter_(1, v_star, 1)
                q_star = q_grad_indices[:, :topv]
                q_mask_pre.scatter_(1, q_star, 1)

            # Always keep CLS token (at index 0)
            v_mask_pre[:, 0] = 1
            q_mask_pre[:, 0] = 1
            sub_attention_mask = attention_mask[start_idx:end_idx]
            q_mask_pre = q_mask_pre * sub_attention_mask
            # Initialize v_mask with fill function
            v_mask = torch.ones_like(v_mask_pre)
            v_mask.masked_fill_(v_mask_pre == 0, 0)  

            # Compute v_mask for each sample
            for b_idx in range(current_sub_batch_size):
                # Get indices of regions where v_mask_pre is 1 for current sample
                v_mask_pre_indices = (v_mask_pre[b_idx] == 1).nonzero(as_tuple=False).squeeze(-1)
                
                # If no preselected regions, skip current sample
                if v_mask_pre_indices.numel() == 0:
                    continue
                    
                # Compute feature contribution (only consider regions where v_mask_pre is 1)
                # Note shape: v_feats is [B, N, D], visual_grad is [B, N, D]
                # We need to sum over dim=2 (feature dimension)
                feature_contribution = torch.sum(sub_v_feats[b_idx] * sub_visual_grad[b_idx], dim=1)  # [num_patches]
                feature_contribution_masked = feature_contribution[v_mask_pre_indices]
                
                # Sort by contribution value
                _, sorted_idx = feature_contribution_masked.sort(descending=True)
                
                # Select top 35% as pathological regions (at least 1)
                topk = max(1, int(len(v_mask_pre_indices) * ture_topk_ratio))
                topk_indices = v_mask_pre_indices[sorted_idx[:topk]]
                
                # The rest of v_mask_pre==1 regions are set to -1
                remaining_indices = v_mask_pre_indices[sorted_idx[topk:]]
                
                # Set v_mask: pathological regions to 1, other v_mask_pre==1 regions to -1
                v_mask[b_idx, topk_indices] = 1.0
                v_mask[b_idx, remaining_indices] = -1.0
                
            # Ensure CLS token is always kept
            v_mask[:, 0] = 1.0

            v_mask_pre = torch.cat([v_mask_pre, torch.ones_like(v_mask_pre)], dim=0)
            v_mask = torch.cat([v_mask, torch.ones_like(v_mask)], dim=0)
            q_mask_pre = torch.cat([q_mask_pre, sub_attention_mask], dim=0)

            # Compute causal reasoning loss  
            sub_logits = model(
                sub_data['combined_images'],
                sub_data['questions_ids'],
                sub_data['attention_mask'],
                sub_data['do_questions_ids'],
                sub_data['do_attention_mask'],
                ae_images=ae_images,
                maml_images=maml_images,
                v_mask_pre=v_mask_pre,
                v_mask=v_mask,
                q_mask_pre=q_mask_pre,
                pattern_embedding=pattern_embedding,
                entity_embedding=entity_embedding,
                epoch=epoch,
                causal_start_epoch=causal_start_epoch,
                training=True
            )
            # 5. Train model
            sub_loss, sub_batch_score = binary_ce_with_hard_negative(sub_logits, targets_all)
            total_cls_loss += sub_loss.item() * current_sub_batch_size
            sub_loss.backward()
            if grad_clip:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            
            total_current += current_sub_batch_size
            total_score += sub_batch_score.sum().item()

        if (i + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"| Batch {i + 1}/{len(data_loader)} | {elapsed * 1000 / log_interval:.2f} ms/batch | "
                f"Causal Feature Decomposition Consistency Loss {total_factor_loss / total_current:.4f} | "
                f"Total Loss {total_cls_loss / total_current:.4f} | "
                f"Score {total_score / total_current * 100:.2f}% | "
                f"LR {lr:.6f}| Phase: Causal Reasoning (sub-batch size: {sub_batch_size})"
            )
            start_time = time.time()
    scheduler.step()
    # Compute average loss and accuracy
    avg_factor_loss = total_factor_loss / max(1, total_current)
    avg_total_loss = total_cls_loss / max(1, total_current)
    avg_score = total_score / max(1, total_current) * 100
    
    if enable_causal:
        avg_factor_loss = total_factor_loss / max(1, total_current)
        logger.info(
            f"Epoch {epoch} Finished | Decoupled Contrastive Consistency Loss {avg_factor_loss:.4f} | Total Loss {avg_total_loss:.4f} | Score {avg_score:.2f}%"
        )
    else:
        logger.info(
            f"Epoch {epoch} Finished | Loss {avg_total_loss:.4f} | Score {avg_score:.2f}%"
        )
    
    # Return detailed loss info for logging
    loss_info = {
        'epoch': epoch,
        'total_cls_loss': avg_total_loss,
        'total_factor_loss': avg_factor_loss if enable_causal else 0.0,
        'total_loss': avg_total_loss + (avg_factor_loss if enable_causal else 0.0),
        'accuracy': avg_score,
        'enable_causal': enable_causal,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    
    return avg_total_loss, avg_score, loss_info


def validate(model, data_loader, criterion, device):
    """Evaluate model performance on validation set"""
    model.eval()
    total_loss = 0.0
    total_causal_loss = 0.0
    total_score = 0.0  # Standard model score
    total_causal_score = 0.0  # Causal reasoning score
    total_contrastive_loss = 0.0
    total_samples = 0

    logger.info("Start validation...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            # =========== 1) Prepare input ===========
            data = prepare_batch_data(batch, device, duplicate_text=False)
            images = data['images']
            questions = data['questions_ids']
            attention_mask = data['attention_mask']
            do_questions = data['do_questions_ids']
            do_attention_mask = data['do_attention_mask']
            targets = data['targets']
            pattern_embedding = data.get('pattern_embedding', None)
            entity_embedding = data.get('entity_embedding', None)
            ae_images = data.get('ae_images', None).to(device)
            maml_images = data.get('maml_images', None).to(device)

            # =========== 2) Forward ===========
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


            loss = criterion(logits, targets)

            batch_score = compute_score_with_logits(logits, targets).sum().item()

            # =========== 4) Statistics ===========
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_score += batch_score
            total_samples += batch_size

    # Compute average metrics
    avg_loss = total_loss / total_samples
    avg_score = total_score / total_samples * 100  # convert to percentage

    logger.info(f"Validation result: Loss {avg_loss:.4f} | Score {avg_score:.2f}%")

    return avg_loss, avg_score
