#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy
import os
import random
import time
import logging
from typing import Literal, Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

from sklearn.mixture import GaussianMixture

from transformers import BertModel, get_linear_schedule_with_warmup


# In[ ]:


# --- CẤU HÌNH LOGGING ---
logging.basicConfig(
    level=logging.INFO, # Ghi lại các log từ mức INFO trở lên (INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s [%(levelname)s] - %(message)s', # Định dạng của mỗi dòng log
    handlers=[
        logging.FileHandler("training.log", mode='w'), # Ghi ra file tên là training.log, mode='w' để ghi mới mỗi lần chạy
        logging.StreamHandler()                        # Gửi log ra màn hình (console)
    ]
)


# In[ ]:


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class _TrainDataset(Dataset):
    """
    Dataset train chung cho image/text.
    - Image: trả Tensor (C,H,W) sau Resize & Normalize (ImageNet).
    - Text: trả string thô; tokenize ở collate_fn để padding theo batch.
    Trả về: (x, y_noisy, index)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        data_column: str,
        noisy_labels: np.ndarray,
        data_type: str,
        image_dir: Optional[str] = None,
        image_size: int = 224,
    ):
        assert data_type in {"image", "text"}, "data_type phải là 'image' hoặc 'text'"
        self.df = df.reset_index(drop=True)
        self.data_column = data_column
        self.noisy_labels = noisy_labels.astype(np.int64)
        self.data_type = data_type
        self.image_dir = image_dir
        self.image_size = image_size

        if self.data_type == "image":
            if not self.image_dir:
                raise ValueError("image_dir là bắt buộc khi data_type='image'.")
            self.transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),  # không augmentation, chỉ resize cố định
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = None  # tokenize ở collate_fn

        if len(self.df) != len(self.noisy_labels):
            raise ValueError(f"Số dòng CSV ({len(self.df)}) khác số dòng feather/noisy ({len(self.noisy_labels)}).")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, fname: str) -> torch.Tensor:
        path = os.path.join(self.image_dir, fname)
        with Image.open(path) as im:
            im = im.convert("RGB")
        return self.transform(im)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if self.data_type == "image":
            x = self._load_image(str(row[self.data_column]))
        else:
            # text thô; collate_fn sẽ tokenize
            x = str(row[self.data_column])
        y = int(self.noisy_labels[idx])  # nhãn nhiễu để train (đã là 0..C-1 theo yêu cầu)
        return x, torch.tensor(y, dtype=torch.long), idx


def _make_text_collate_fn(max_length: int = 512, pretrained_name: str = "bert-base-uncased"):
    """
    Collate cho text: tokenize theo batch -> dict tensors (input_ids, attention_mask, token_type_ids).
    Trả về: (inputs_dict, labels, indices)
    """
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_name)

    def collate(batch: List[Tuple[str, torch.Tensor, int]]):
        texts = [b[0] for b in batch]
        labels = torch.stack([b[1] for b in batch], dim=0)
        indices = torch.tensor([b[2] for b in batch], dtype=torch.long)
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # đảm bảo có token_type_ids (BERT sử dụng)
        if "token_type_ids" not in tokenized:
            tokenized["token_type_ids"] = torch.zeros_like(tokenized["input_ids"])
        return dict(tokenized), labels, indices

    return collate


class TrainDataLoader:
    """
    Dataloader chỉ cho TRAIN.

    Args:
        csv_path: đường dẫn CSV (cột data & cột label sạch).
        feather_path: file feather có cột 'label' = nhãn NHIỄU (0..C-1), cùng thứ tự với CSV.
        data_column: tên cột dữ liệu (text hoặc tên file ảnh) trong CSV.
        label_column: tên cột nhãn sạch trong CSV (0..C-1).
        image_dir: thư mục chứa ảnh (bắt buộc nếu data_type='image').
        data_type: 'image' hoặc 'text'.
        batch_size: kích thước batch.
        num_workers: số worker cho DataLoader (mặc định 4).
        image_size: kích thước resize ảnh (mặc định 224).
        text_max_length: max_length khi tokenize BERT (mặc định 512).
    """
    def __init__(
        self,
        csv_path: str,
        feather_path: str,
        data_column: str,
        label_column: str,
        image_dir: Optional[str],
        data_type: str,
        batch_size: int,
        num_workers: int = 4,
        image_size: int = 224,
        text_max_length: int = 512,
    ):
        self.csv_path = csv_path
        self.feather_path = feather_path
        self.data_column = data_column
        self.label_column = label_column
        self.image_dir = image_dir
        self.data_type = data_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.text_max_length = text_max_length

        # Đọc dữ liệu
        df = pd.read_csv(self.csv_path)
        if self.data_column not in df.columns or self.label_column not in df.columns:
            raise ValueError(f"CSV phải có cột '{self.data_column}' và '{self.label_column}'.")
        fdf = pd.read_feather(self.feather_path)
        if "label" not in fdf.columns:
            raise ValueError("Feather phải có cột 'label' (nhãn nhiễu).")

        # Lấy clean/noisy labels (đã 0..C-1 theo yêu cầu)
        self.clean_labels = df[self.label_column].to_numpy(dtype=np.int64)
        self.noisy_labels = fdf["label"].to_numpy(dtype=np.int64)

        # Dataset
        self.train_dataset = _TrainDataset(
            df=df,
            data_column=self.data_column,
            noisy_labels=self.noisy_labels,
            data_type=self.data_type,
            image_dir=self.image_dir,
            image_size=self.image_size,
        )

        # Collate cho text
        collate = _make_text_collate_fn(max_length=self.text_max_length) if self.data_type == "text" else None

        pin_mem = torch.cuda.is_available()
        self.trainloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_mem,
            persistent_workers=(self.num_workers > 0),
            collate_fn=collate,
        )

    def run(self) -> Tuple[DataLoader, np.ndarray, np.ndarray]:
        """Trả: (trainloader, noisy_labels, clean_labels)."""
        return self.trainloader, self.noisy_labels, self.clean_labels


# In[ ]:


'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)  # number 1 indicates how many channels
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # [CHANGE] dùng adaptive pool để hỗ trợ mọi input size (32, 96, 224, ...)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.c_linear = nn.Linear(512 * block.expansion, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        # [CHANGE] an toàn biến trả về khi lout <= 4
        feature = None
        out_c = None

        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            # [CHANGE] thay vì F.avg_pool2d(out, 4)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            feature = out
            out_c = self.c_linear(out)
            out = self.linear(out)
        return out, feature, out_c


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


# In[ ]:


class BertTextClassifier(nn.Module):
    """
    BERT -> Dropout -> Linear(num_classes)
    Trả về: logits (B, C), feature (B, H), out_c (B, 1)
    - logits: dùng cho loss CE (multi-class)
    - feature: embedding đã qua dropout, trước classifier
    - out_c: logit 1 chiều (giữ giao diện với pipeline hiện tại)
    """
    def __init__(
        self,
        num_classes: int,
        pretrained_name: str = "bert-base-uncased",
        dropout: float = 0.3,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)
        self.c_linear = nn.Linear(hidden, 1)  # giữ pipeline: trả thêm out_c

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, inputs, lin: int = 0, lout: int = 2):
        """
        inputs: dict có các keys 'input_ids', 'attention_mask', (tuỳ tokenizer có thể có 'token_type_ids')
        lin/lout để giữ chữ ký giống ResNet; ở đây không cắt tầng nên bỏ qua.
        """
        # Đảm bảo có token_type_ids (một số tokenizer có thể không tạo)
        if isinstance(inputs, dict) and "token_type_ids" not in inputs:
            inputs = dict(inputs)  # clone shallow để không đụng batch gốc
            inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])

        outputs = self.bert(**inputs, return_dict=True)
        # pooled_output có sẵn (tanh( W * CLS )), nếu vì lý do nào đó None thì fallback về CLS token
        if outputs.pooler_output is not None:
            pooled = outputs.pooler_output  # (B, H)
        else:
            pooled = outputs.last_hidden_state[:, 0]  # (B, H) lấy token [CLS]

        feat = self.dropout(pooled)      # feature dùng để phân loại
        out_c = self.c_linear(feat)      # (B, 1) - để khớp pipeline
        logits = self.classifier(feat)   # (B, num_classes)
        return logits, feat, out_c


# In[ ]:


# ----- helper: per-sample losses -----
@torch.no_grad()
def _per_sample_losses(
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    data_type: Literal["image", "text"],
    num_samples: int,
    show_tqdm: bool = True,
    use_amp: bool = False, # [AMP] Thêm tham số để bật/tắt autocast
) -> np.ndarray:
    model.eval()
    losses = torch.empty(num_samples, dtype=torch.float32, device=device)
    it = tqdm(trainloader, desc="Eval per-sample loss", unit="batch", leave=False, disable=not show_tqdm)
    for batch in it:
        if data_type == "text":
            inputs, labels, index = batch
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        else:
            inputs, labels, index = batch
            inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        index  = index.to(device, non_blocking=True)

        # [AMP] Bọc forward pass với autocast để tăng tốc tính toán trên GPU
        with autocast("cuda", enabled=use_amp):
            logits, _, _ = model(inputs)
            loss_vec = F.cross_entropy(logits, labels, reduction="none")

        losses[index] = loss_vec
    return losses.detach().cpu().numpy()

# ----- [CHANGE] normalize losses per epoch -----
def _normalize_losses(
    losses: np.ndarray,
    method: Literal["minmax", "robust", "none"] = "minmax",
    eps: float = 1e-8,
) -> np.ndarray:
    """Chuẩn hoá loss theo epoch để M1 so sánh được giữa các epoch."""
    x = losses.astype(np.float64)
    # loại bỏ inf/nan về giá trị hữu hạn an toàn
    x = np.nan_to_num(x, nan=0.0, posinf=np.max(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0, neginf=0.0)

    if method == "none":
        return x

    if method == "minmax":
        lo, hi = np.min(x), np.max(x)
        denom = max(hi - lo, eps)
        z = (x - lo) / denom
        return np.clip(z, 0.0, 1.0)

    # robust z-score theo median/MAD (đơn vị z, không ép 0..1)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    denom = max(mad * 1.4826, eps)  # 1.4826 ~ chuyển MAD -> sigma
    return (x - med) / denom

# ----- helper: M1 from losses -----
def _m1_from_losses(losses: np.ndarray, random_state: int = 42) -> float:
    x = losses.reshape(-1, 1).astype(np.float64)
    # (sau normalize, x đã hữu hạn)
    if x.shape[0] < 3:
        return 0.0
    gmm = GaussianMixture(n_components=2, random_state=random_state, covariance_type="full")
    gmm.fit(x)
    mu = np.sort(gmm.means_.flatten())
    return float(abs(mu[1] - mu[0]))

# ----- main: estimate es by M1 -----
def estimate_es_m1(
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    data_type: Literal["image", "text"],
    *,
    max_scan_epochs: int = 60,
    lr: float = 2e-2,
    optimizer_name: Literal["SGD", "AdamW"] = "SGD",
    weight_decay: float = 1e-3,
    momentum: float = 0.9,
    random_state: int = 42,
    patience: Optional[int] = None,
    clone_model: bool = True,
    show_tqdm: bool = True,
    normalize: Literal["minmax", "robust", "none"] = "minmax",
    use_amp: bool = True, # [AMP] Thêm tùy chọn để bật/tắt AMP
) -> Tuple[int, List[float]]:
    work_model = copy.deepcopy(model) if clone_model else model
    work_model.to(device)
    if optimizer_name == "SGD":
        opt = torch.optim.SGD(work_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(work_model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = torch.nn.CrossEntropyLoss()

    # [AMP] Chỉ bật AMP khi có GPU và được người dùng cho phép
    is_amp_enabled = use_amp and device.type == "cuda"
    if is_amp_enabled:
        # [AMP] Khởi tạo GradScaler để quản lý việc scale gradient
        scaler = GradScaler("cuda", enabled=is_amp_enabled)

    num_samples = len(trainloader.dataset)
    best_epoch, best_m1, hist_m1, no_imp = 1, -float("inf"), [], 0

    epoch_iter = tqdm(range(1, max_scan_epochs + 1), desc="Scan epochs (M1)", unit="epoch",
                      disable=not show_tqdm)
    for epoch in epoch_iter:
        work_model.train()
        train_it = tqdm(trainloader, desc=f"Train e{epoch}", unit="batch", leave=False, disable=not show_tqdm)
        for batch in train_it:
            if data_type == "text":
                inputs, labels, _ = batch
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            else:
                inputs, labels, _ = batch
                inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # [AMP] Bọc forward pass và tính loss bằng autocast
            with autocast("cuda", enabled=is_amp_enabled):
                logits, _, _ = work_model(inputs)
                loss = ce(logits, labels)

            opt.zero_grad(set_to_none=True)

            # [AMP] Sử dụng scaler để thực hiện backward và step nếu AMP được bật
            if is_amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

        # [AMP] Truyền cờ is_amp_enabled vào hàm tính loss
        losses_np = _per_sample_losses(work_model, trainloader, device, data_type, num_samples, show_tqdm=show_tqdm, use_amp=is_amp_enabled)
        losses_np = _normalize_losses(losses_np, method=normalize)
        m1 = _m1_from_losses(losses_np, random_state=random_state)
        hist_m1.append(m1)
        logging.info(f"[Scan][Epoch {epoch}] M1={m1:.6f}")
        epoch_iter.set_postfix_str(f"M1={m1:.6f}")

        if m1 > best_m1:
            best_m1, best_epoch, no_imp = m1, epoch, 0
        else:
            no_imp += 1
            if patience is not None and no_imp >= patience:
                logging.info("Early stop triggered!")
                epoch_iter.set_postfix_str(f"M1={m1:.6f} (early stop)")
                break

    logging.info(f"=> estimated_es (M1) = {best_epoch}")
    return best_epoch, hist_m1


# In[ ]:


# ----- Lớp Loss của SELC (Giữ nguyên logic gốc) -----
class SELCLoss(nn.Module):
    def __init__(self, labels, num_classes, es=10, momentum=0.9, device='cuda'):
        super(SELCLoss, self).__init__()
        self.num_classes = num_classes
        self.soft_labels = torch.zeros(len(labels), num_classes, dtype=torch.float).to(device)
        self.soft_labels[torch.arange(len(labels)), labels] = 1
        self.es = es
        self.momentum = momentum
        self.CEloss = nn.CrossEntropyLoss()

    def forward(self, logits, labels, index, epoch):
        pred = F.softmax(logits, dim=1)
        if epoch <= self.es:
            ce = self.CEloss(logits, labels)
            return ce
        else:
            pred_detach = F.softmax(logits.detach(), dim=1)
            self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * pred_detach

            selc_loss = -torch.sum(torch.log(pred) * self.soft_labels[index], dim=1)
            return selc_loss.mean()


# In[ ]:


if __name__ == '__main__':
    # [CONFIG] Thay đổi các giá trị trong dict này để chạy với dataset khác
    config = {
        "dataset_name": "CIFAR-10_sym40",
        "csv_path": "./Data/Cifar10-test/Cifar10-test.csv",
        "feather_path": "./Data/Cifar10-test/cifar10-test-clip-b16-noise/cifar10-test_sym40.feather",
        "data_column": "image_name",
        "label_column": "label",
        "image_dir": "./Data/Cifar10-test/images",
        "data_type": "image",      # 'image' hoặc 'text'
        "batch_size": 128,         # Mặc định từ src gốc
        "num_workers": 4,          # Tham số num_workers
        "num_epochs": 200,         # Mặc định từ src gốc
        "es": None,                  # Đặt None để tự động ước tính, hoặc điền số nguyên (vd: 40)
        "alpha": 0.9,              # Mặc định từ src gốc
        "log_interval": 30,
        "seed": 42,
        "max_duration_seconds": None # Thời gian tối đa thực thi, đặt None nếu không giới hạn
    }

    # ----- Tự động thiết lập siêu tham số -----
    if config["data_type"] == 'image':
        hparams = {"lr": 0.02, "op": "SGD", "lr_s": "MultiStepLR"}
    else: # text
        hparams = {"lr": 2e-5, "op": "AdamW", "lr_s": "LinearWarmup"}
    logging.info(f"Running with config: {config['dataset_name']}")
    logging.info(f"Hyperparameters: {hparams}")

    # ----- Thiết lập môi trường -----
    start_time = time.time()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    # ----- Tải dữ liệu -----
    loader = TrainDataLoader(
        csv_path=config["csv_path"], feather_path=config["feather_path"],
        data_column=config["data_column"], label_column=config["label_column"],
        image_dir=config["image_dir"], data_type=config["data_type"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )
    trainloader, noisy_labels, clean_labels = loader.run()
    num_classes = int(np.max(clean_labels)) + 1

    # ----- Khởi tạo Model  -----
    if config["data_type"] == 'image':
        model = ResNet34(num_classes=num_classes)
    else:
        model = BertTextClassifier(num_classes=num_classes)

    # [DataParallel] Tự động sử dụng nhiều GPU nếu có
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    model.to(device)

    # ----- Ước tính Turning Point (nếu cần) -----
    es = config["es"]
    if es is None:
        logging.info("`es` is None, starting automatic turning point estimation...")
        scan_lr = hparams["lr"]
        scan_op = hparams["op"]

        estimated_es_val, _ = estimate_es_m1(
            model=model, trainloader=trainloader, device=device, data_type=config["data_type"],
            max_scan_epochs=60, lr=scan_lr, optimizer_name=scan_op, weight_decay=1e-3,
            momentum=0.9, random_state=config["seed"], patience=12, clone_model=True,
            show_tqdm=True, normalize="minmax", use_amp=use_amp
        )
        # [ES] Áp dụng công thức Te = T - 10 từ paper
        es = max(1, estimated_es_val - 10)
        # es = estimated_es_val
        logging.info(f"Automatic estimation finished. Using es = {es} (T={estimated_es_val} - 10)")
    else:
        logging.info(f"Using predefined es = {es}")

    # ----- Khởi tạo Optimizer, Scheduler và Loss -----
    if hparams["op"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=hparams["lr"], momentum=0.9, weight_decay=1e-3)
    else: # AdamW
        optimizer = optim.AdamW(model.parameters(), lr=hparams["lr"], weight_decay=0.01)

    if hparams["lr_s"] == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
    else: # LinearWarmup
        num_training_steps = len(trainloader) * config["num_epochs"]
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    criterion = SELCLoss(noisy_labels, num_classes, es, config["alpha"], device)
    scaler = GradScaler("cuda", enabled=use_amp)

    # ----- Vòng lặp huấn luyện chính -----
    logging.info(f"\nStarting main training for {config['num_epochs']} epochs...")

    epoch_iter_main = tqdm(range(1, config["num_epochs"] + 1), desc="Main Training", unit="epoch")
    for epoch in epoch_iter_main:
        # [TIME LIMIT] Kiểm tra thời gian ở đầu mỗi epoch
        if config["max_duration_seconds"] is not None:
            elapsed_time = time.time() - start_time
            if elapsed_time >= config["max_duration_seconds"]:
                logging.info(f"\n[TIME LIMIT] Đã đạt giới hạn thời gian {config['max_duration_seconds']} giây. Dừng huấn luyện.")
                break # Thoát khỏi vòng lặp training
            else:
                logging.info(f"\n[Time] Run for: {elapsed_time} giây")

        model.train()

        batch_iter = tqdm(trainloader, desc=f"Train Epoch {epoch}", unit="batch", leave=False)
        total_loss = 0.0

        for batch_idx, batch in enumerate(batch_iter):
            if config["data_type"] == "text":
                inputs, target, index = batch[0], batch[1], batch[2]
                inputs = {k: v.to(device, non_blocking=True) for k,v in inputs.items()}
            else:
                inputs, target, index = batch[0], batch[1], batch[2]
                inputs = inputs.to(device, non_blocking=True)
            target, index = target.to(device, non_blocking=True), index.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Thực hiện training step
            with autocast("cuda", enabled=use_amp):
                output, _, _ = model(inputs)
                loss = criterion(output, target, index, epoch)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # [SCHEDULER] Cập nhật mỗi BƯỚC cho LinearWarmup
            if hparams["lr_s"] == "LinearWarmup":
                scheduler.step()

            total_loss += loss.item()

            # In log theo interval
            if batch_idx % config['log_interval'] == 0:
                # Giữ lại định dạng logging.info của file gốc
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(target), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))

            current_lr = optimizer.param_groups[0]['lr']
            batch_iter.set_postfix_str(f"Loss: {loss.item():.4f} LR: {current_lr:.6f}")

        # [SCHEDULER] Cập nhật mỗi EPOCH cho MultiStepLR
        if hparams["lr_s"] == "MultiStepLR":
            scheduler.step()

        # Cập nhật thông tin loss trung bình lên thanh tqdm của epoch
        avg_loss_epoch = total_loss / len(trainloader)
        logging.info(f"--- End of Epoch {epoch}/{config['num_epochs']} --- Average Loss: {avg_loss_epoch:.4f} ---")
        epoch_iter_main.set_postfix_str(f"Avg Loss: {avg_loss_epoch:.4f}")

    logging.info("Training finished.")

    # ----- Lưu model -----
    output_dir = f"./output_{config['dataset_name']}_es{es}_seed{config['seed']}"
    os.makedirs(output_dir, exist_ok=True)

    final_model_path = os.path.join(output_dir, "final_model.pth")
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")

    # --- Calculate metric and save ---
    logging.info("\nCalculating final correction metrics...")

    _, corrected_labels_tensor = torch.max(criterion.soft_labels, dim=1)
    corrected_labels = corrected_labels_tensor.cpu().numpy()

    # 1. Tính Correction Precision: Trong số các nhãn đã được thay đổi, bao nhiêu % được sửa đúng?
    changed_mask = (noisy_labels != corrected_labels)
    total_changed = np.sum(changed_mask)

    correctly_fixed_mask = (corrected_labels == clean_labels)
    correctly_changed_count = np.sum(changed_mask & correctly_fixed_mask)
    precision = (correctly_changed_count / total_changed * 100) if total_changed > 0 else 0.0

    # 2. Tính Error Rate: Tỷ lệ nhãn sai sau khi đã sửa
    final_errors = np.sum(corrected_labels != clean_labels)
    error_rate = final_errors / len(clean_labels) * 100

    # In các chỉ số ra màn hình
    logging.info(f"Total labels changed: {total_changed}/{len(clean_labels)}")
    logging.info(f"Correction Precision: {correctly_changed_count}/{total_changed} = {precision:.2f}%")
    logging.info(f"Final Error Rate: {final_errors}/{len(clean_labels)} = {error_rate:.2f}%")

    # Save correct info
    results_df = pd.DataFrame({
        'Index': range(len(clean_labels)),
        'noisy_label': noisy_labels,
        'fixed_label': corrected_labels,
        'true_label': clean_labels
    })
    base_feather_name = os.path.basename(config["feather_path"])
    file_stem = os.path.splitext(base_feather_name)[0]
    new_csv_filename = f"{file_stem}_correction_results.csv"
    csv_path = os.path.join(output_dir, new_csv_filename)
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Detailed correction results saved to {csv_path}")


# In[ ]:




