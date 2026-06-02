"""
v71 추론을 위한 최소 스펙트럼 기반 모델.

인코더(SpectralEncoderV2)와 위치 인코딩(SingleFrequencySpectralEmbedding)만 포함
"""

import torch
import torch.nn as nn
import math


class SingleFrequencySpectralEmbedding(nn.Module):
    """
    사인파 인코딩을 이용한 파장 인식 위치 임베딩.

    파장을 위치 유사 값 [0, max_seq_len]으로 정규화한 후
    표준 사인파 위치 인코딩을 적용합니다.
    """
    def __init__(self, embed_dim, max_wavelength=2500.0, min_wavelength=350.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim은 짝수여야 합니다.")
        self.embed_dim = embed_dim
        self.max_wavelength = max_wavelength
        self.min_wavelength = min_wavelength

    def forward(self, wv_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wv_flat: nm 단위의 파장 값, 형상 (seq_len,) 또는 (batch*seq_len,)
        Returns:
            위치 임베딩, 형상 (seq_len, embed_dim) 또는 (batch*seq_len, embed_dim)
        """
        wv_normalized = (wv_flat - self.min_wavelength) / (self.max_wavelength - self.min_wavelength + 1e-8)
        wv_normalized = wv_normalized.clamp(0, 1)

        # 유효 시퀀스 길이로 스케일링 (표준 트랜스포머 관행)
        # 수치 안정성을 위해 float32로 강제 변환
        position = (wv_normalized * 10000.0).to(torch.float32)

        half = self.embed_dim // 2
        idx = torch.arange(half, device=wv_flat.device, dtype=torch.float32)

        # FP16 오버플로 방지를 위해 로그 공간 사용
        log_div_term = (2.0 * idx / self.embed_dim) * math.log(10000.0)
        div_term = torch.exp(log_div_term)

        angles = position.unsqueeze(1) / div_term.unsqueeze(0)

        sin_part = torch.sin(angles)
        cos_part = torch.cos(angles)

        out = torch.stack((sin_part, cos_part), dim=-1)  # (seq_len, E/2, 2)
        return out.view(-1, self.embed_dim).to(wv_flat.dtype)


class SpectralEncoderV2(nn.Module):
    """
    초분광 데이터를 위한 스펙트럼 트랜스포머 인코더.

    1차원 스펙트럼 [B, band_dim]을 입력받아 파장 인식 위치 인코딩 +
    트랜스포머 인코더 + 마스크 평균 풀링을 통해
    고정 크기 표현 [B, embed_dim]을 출력합니다.
    """
    def __init__(self, embed_dim=128, depth=4, heads=4, wavelengths=None,
                 dim_feedforward=None, band_dim=None,
                 dropout=0.1, activation='relu'):
        """
        Args:
            embed_dim: 임베딩 차원
            depth: 트랜스포머 레이어 수
            heads: 어텐션 헤드 수
            wavelengths: 위치 인코딩을 위한 파장 값 (nm)
            dim_feedforward: FFN 은닉층 차원 (기본값: embed_dim * 4)
            band_dim: 스펙트럼 밴드 수
            dropout: 드롭아웃 비율
            activation: 활성화 함수 ('relu' 또는 'gelu')
        """
        super().__init__()

        if band_dim is not None:
            self.band_dim = band_dim
        elif wavelengths is not None:
            self.band_dim = len(wavelengths)
        else:
            self.band_dim = 80
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = heads
        self.wavelengths = wavelengths

        self.embedding = nn.Linear(1, embed_dim)
        self.pos_embedding = SingleFrequencySpectralEmbedding(embed_dim)

        if dim_feedforward is None:
            dim_feedforward = embed_dim * 4
        dim_feedforward = int(dim_feedforward)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=activation
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def use_checkpointing(self):
        """메모리 사용량 감소를 위한 그래디언트 체크포인팅 활성화."""
        for layer in self.transformer.layers:
            layer.use_checkpointing = True
        return self

    def forward(self, x, src_key_padding_mask=None, selected_indices=None, dataset_indices=None):
        """
        Args:
            x: 입력 스펙트럼 [B, band_dim]
            src_key_padding_mask: 불리언 마스크 [B, band_dim], True = 패딩/무시
            selected_indices: 샘플별 밴드 인덱스 [B, band_dim] (선택사항)
            dataset_indices: 무시됨 (API 호환성을 위해 유지)
        Returns:
            풀링된 표현 [B, embed_dim]
        """
        B, D = x.shape
        assert D == self.band_dim, f"밴드 수 {self.band_dim}개를 기대했으나 {D}개를 받았습니다"

        # 각 밴드 값을 독립적으로 임베딩
        x = x.unsqueeze(-1)  # [B, band_dim, 1]
        x = self.embedding(x)  # [B, band_dim, embed_dim]

        # 파장 인식 위치 인코딩
        if self.wavelengths is not None:
            if selected_indices is not None:
                wavelengths_on_device = self.wavelengths.to(x.device)
                pos_enc_list = []
                for i in range(B):
                    sample_wavelengths = wavelengths_on_device[selected_indices[i]]
                    pos_enc_list.append(self.pos_embedding(sample_wavelengths))
                pos_enc = torch.stack(pos_enc_list, dim=0).to(x.dtype)
                x = x + pos_enc
            else:
                wavelengths_used = self.wavelengths[:self.band_dim].to(x.device)
                pos_enc = self.pos_embedding(wavelengths_used)
                pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1).to(x.dtype)
                x = x + pos_enc
        else:
            raise RuntimeError("위치 인코딩을 위해 파장 정보가 필요합니다.")

        # 트랜스포머
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 마스크 평균 풀링
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).float().unsqueeze(-1)  # [B, band_dim, 1]
            masked_x = x * valid_mask
            sum_x = masked_x.sum(dim=1)
            count_valid = valid_mask.sum(dim=1).clamp(min=1.0)
            return sum_x / count_valid
        else:
            return x.mean(dim=1)
