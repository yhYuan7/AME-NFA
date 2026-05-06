"""
Core Model: Adaptive Modality-aware Embedding (AME) with Noise-robust Feature Aggregation (NFA).
Corresponds to the methodology in Section III of the paper.
"""
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x, ratio=1.0):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:x.size(1), :].transpose(0, 1) * ratio
        return x


class GatedMoE(nn.Module):
    """Gated Mixture-of-Experts for multi-feature fusion."""
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(input_dim * num_experts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, expert_list):
        # expert_list: List of [B, D]
        expert_tensor = torch.stack(expert_list, dim=1)  # [B, N, D]
        B, N, D = expert_tensor.size()
        flat_input = expert_tensor.reshape(B, N * D)     # [B, N*D]
        weights = self.gating(flat_input)                # [B, N]
        weights = weights.unsqueeze(-1)                  # [B, N, 1]
        fused = torch.sum(weights * expert_tensor, dim=1)  # [B, D]
        return fused


class AMENFA(nn.Module):
    """
    Adaptive Modality-aware Embedding with Noise-robust Feature Aggregation.
    Supports multi-feature projection, modal tokens, positional encoding,
    and Transformer-based aggregation.
    """
    def __init__(
        self,
        feature_names,
        input_dim_trans=128,
        num_head=8,
        dim_feedforward=2048,
        nlayers=2,
        dropout=0.3,
        modal_token_std=0.02,
        use_final_project_layer=True,
        use_pos_emb=True,
        embed_dim=128
    ):
        super().__init__()
        self.input_dim_trans = input_dim_trans
        self.use_final_project_layer = use_final_project_layer

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=input_dim_trans,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Modal token embedding
        self.use_modal_token = modal_token_std > 0
        if self.use_modal_token:
            def _init_weights(module):
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=modal_token_std)
            self.type_embedding = nn.Embedding(2, input_dim_trans)
            self.type_embedding.apply(_init_weights)

        # Positional encoding
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_encoder = PositionalEncoding(input_dim_trans)
            self.pos_ratio = 1e-5

        # Feature projection layers
        voice_features = [f for f in feature_names if f.startswith("v")]
        face_features = [f for f in feature_names if f.startswith("f")]
        self.face_features_count = len(face_features)

        self._create_project_layers("v", voice_features)
        self._create_project_layers("f", face_features)

        # Final projection to embedding space
        self.face_project_layer = self._get_final_project_layer(input_dim_trans, embed_dim)
        self.voice_project_layer = self._get_final_project_layer(input_dim_trans, embed_dim)

        # MoE fusion
        self.voice_moe = GatedMoE(input_dim_trans, 64, len(voice_features))
        self.face_moe = GatedMoE(input_dim_trans, 64, len(face_features))

    def _get_final_project_layer(self, input_size, output_size):
        if self.use_final_project_layer:
            return nn.Sequential(nn.Linear(input_size, output_size))
        return nn.Identity()

    def _create_project_layers(self, name, feature_list):
        for i, feat_name in enumerate(feature_list):
            raw_dim = int(feat_name.split("_")[-1])
            layer = nn.Sequential(
                nn.Dropout(),
                nn.Linear(raw_dim, self.input_dim_trans)
            )
            setattr(self, f"project_{name}{i}", layer)

    def forward(self, input_list):
        face_count = self.face_features_count
        face = input_list[0:face_count]
        voice = input_list[face_count:]

        voice_emb = self.voice_encoder(voice)
        face_emb = self.face_encoder(face)

        # Ensure 2D
        if face_emb.dim() == 1:
            face_emb = face_emb.unsqueeze(0)
        if voice_emb.dim() == 1:
            voice_emb = voice_emb.unsqueeze(0)

        return voice_emb, face_emb

    def face_encoder(self, input_list):
        emb = self._common_encoder(input_list, "f")
        return self.face_project_layer(emb)

    def voice_encoder(self, input_list):
        emb = self._common_encoder(input_list, "v")
        return self.voice_project_layer(emb)

    def _common_encoder(self, input_list, name):
        assert name in ["v", "f"]
        features = []
        for i, inp in enumerate(input_list):
            layer = getattr(self, f"project_{name}{i}")
            feat = layer(inp)  # [B, D]
            features.append(feat)

        # MoE fusion
        moe_module = self.voice_moe if name == "v" else self.face_moe
        fused = moe_module(features)  # [B, D]

        # Add modal token
        if self.use_modal_token:
            idx = 0 if name == "v" else 1
            batch_size = input_list[0].shape[0]
            device = input_list[0].device
            emb_token = self.type_embedding(torch.LongTensor([idx] * batch_size).to(device))
            fused = fused + emb_token

        # Transformer aggregation
        return self._calc_emb(fused.unsqueeze(1))  # [B, 1, D] -> [B, D]

    def _calc_emb(self, input_val):
        if self.use_pos_emb:
            input_val = self.pos_encoder(input_val, self.pos_ratio)
        encoder_result = self.transformer_encoder(input_val)  # [B, 1, D]
        output = encoder_result.mean(dim=1)  # [B, D]
        return output
