import torch
import torch.nn as nn
import math

from utils import MAX_SEQ_LEN


# --- 辅助模块: 位置编码 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# --- 辅助模块: CNN中的基础残差块 ---
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


# --- 主模型: LPR_Transformer ---
class LPR_Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, img_height=48, img_width=160):
        super(LPR_Transformer, self).__init__()

        self.d_model = d_model

        # 1. CNN 视觉编码器 (Visual Encoder)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2 -> 24, 80
            BasicBlock(64, 128, stride=2),  # H/4, W/4 -> 12, 40
            BasicBlock(128, 256, stride=2),  # H/8, W/8 -> 6, 20
            BasicBlock(256, d_model, stride=2),  # H/16, W/16 -> 3, 10
        )
        # 经过CNN后, H和W都缩小了16倍。feature_map_height = 48/16=3, feature_map_width = 160/16=10
        # 序列长度将是 3 * 10 = 30
        self.feature_seq_len = (img_height // 16) * (img_width // 16)

        # 2. Transformer 解码器 (Sequence Decoder)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=MAX_SEQ_LEN)  # MAX_SEQ_LEN = 10

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # 3. 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        """生成一个上三角矩阵的mask，防止decoder看到未来的token"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src_img, tgt_tokens):
        """
        训练时的前向传播
        - src_img: (B, C, H, W) - 输入的批量图像
        - tgt_tokens: (B, L) - 目标字符序列 (ground truth)，例如 [<s>, 皖, A, ..., </s>]
        """
        # --- CNN编码 ---
        # src_img: [B, 3, 48, 160]
        cnn_features = self.cnn_encoder(src_img)  # -> [B, d_model, H', W'] e.g. [B, 512, 3, 10]

        # --- 变形以适应Transformer ---
        B, C, H, W = cnn_features.shape
        # -> [B, d_model, H'*W'] -> [H'*W', B, d_model]
        memory = cnn_features.view(B, C, -1).permute(2, 0, 1)  # -> [30, B, 512]

        # --- Transformer解码 ---
        # 准备目标序列
        tgt_tokens = tgt_tokens.transpose(0, 1)  # -> [L, B]
        tgt_len = tgt_tokens.size(0)

        # 生成目标序列的mask，防止看到未来信息
        device = tgt_tokens.device
        tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(device)

        # 嵌入目标序列并添加位置编码
        tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.d_model)
        tgt_pos = self.pos_encoder(tgt_emb)  # -> [L, B, d_model]

        # 解码
        # decoder_output: [L, B, d_model]
        decoder_output = self.transformer_decoder(tgt_pos, memory, tgt_mask=tgt_mask)

        # 输出层
        # LPR_Rec: [L, B, vocab_size]
        output = self.fc_out(decoder_output)

        # 转换回 Batch-first 格式
        # LPR_Rec: [B, L, vocab_size]
        return output.permute(1, 0, 2)

    def inference(self, src_img, start_token_idx, end_token_idx, max_len=MAX_SEQ_LEN):
        """
        推理时的前向传播 (自回归生成)
        - src_img: (1, C, H, W) - 单张输入图像
        - start_token_idx: "<s>" 的索引
        - end_token_idx: "</s>" 的索引
        """
        self.eval()
        device = src_img.device

        # --- CNN编码 (同上) ---
        cnn_features = self.cnn_encoder(src_img)
        B, C, H, W = cnn_features.shape
        memory = cnn_features.view(B, C, -1).permute(2, 0, 1)  # -> [SeqLen, 1, d_model]

        # ---自回归解码---
        # 初始化目标序列，只包含一个起始符<s>
        # tgt_tokens: [1, 1] (L=1, B=1)
        tgt_tokens = torch.full((1, 1), start_token_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_len = tgt_tokens.size(0)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(device)

            # 嵌入和位置编码
            tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.d_model)
            tgt_pos = self.pos_encoder(tgt_emb)

            # 解码
            output = self.transformer_decoder(tgt_pos, memory, tgt_mask=tgt_mask)

            # 取出最后一个时间步的预测结果
            last_token_logits = self.fc_out(output[-1, :, :])  # -> [1, vocab_size]

            # 找到概率最高的token
            pred_token_idx = last_token_logits.argmax(dim=-1)  # -> [1]

            # 将预测的token拼接到现有序列中，作为下一步的输入
            tgt_tokens = torch.cat([tgt_tokens, pred_token_idx.unsqueeze(0)], dim=0)

            # 如果预测到</s>，则停止生成
            if pred_token_idx.item() == end_token_idx:
                break

        return tgt_tokens.squeeze(1)  # 返回 [L]