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

# Get logger
logger = logging.getLogger(__name__)

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax to get predicted class index
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)  # create one-hot encoding
    scores = (one_hots * labels)  # get label value at predicted class
    return scores

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
    

    for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} Training")):
        # =========== 1) Prepare input ===========
        images = batch['images'].to(device)
        questions_ids = batch["questions"]['input_ids'].to(device)
        attention_mask = batch["questions"]['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        batch_size = questions_ids.size(0)

        # batch_size = data['batch_size']
        # =========== 2) Forward ===========
        # Assume model returns (logits_o, logits_a, decoder, ori_vis_features, aug_vis_features)
        logits = model(
            images,
            questions_ids,
            attention_mask
        )

        # =========== 3) Main loss (classification) ===========
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Note: Consistency loss is not counted, as it is not used in base training
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
    scheduler.step()
    # Calculate average loss and accuracy
    avg_total_loss = total_cls_loss / max(1, total_current)
    avg_score = total_score / max(1, total_current) * 100
    
    logger.info(
        f"Epoch {epoch} Finished | Loss {avg_total_loss:.4f} | Score {avg_score:.2f}%"
    )
    loss_info = {
        'epoch': epoch,
        'total_cls_loss': avg_total_loss,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    return avg_total_loss, avg_score, loss_info

def validate(model, data_loader, criterion, device):
    model.eval()
    total_cls_loss = 0.0
    total_score = 0.0
    total_current = 0
    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc=f"Validating")):
            images = batch['images'].to(device)
            questions_ids = batch["questions"]['input_ids'].to(device)
            attention_mask = batch["questions"]['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            batch_size = questions_ids.size(0)

            logits = model(
                images,
                questions_ids,
                attention_mask
            )

            loss = criterion(logits, targets)
            total_cls_loss += loss.item() * batch_size
            total_score += compute_score_with_logits(logits, targets).sum().item()
            total_current += batch_size
    logger.info(
        f"Validation result: Loss {total_cls_loss / total_current:.4f} | Score {total_score / total_current * 100:.2f}%"
    )
    return total_cls_loss / total_current, total_score / total_current * 100