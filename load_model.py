"""
v71 스펙트럼 기반 모델의 최소 로더.

사용법:
    from load_model import load_encoder

    # 인코더만 로드 (다운스트림 태스크용)
    encoder = load_encoder()          # 기본값: 이 디렉터리의 checkpoint_best.pth
    encoder = load_encoder(band_dim=162)  # 사용자 데이터셋에 맞게 band_dim 재설정

    # 추론 예제
    import torch
    x = torch.randn(4, 80)  # [batch_size, band_dim]
    with torch.no_grad():
        features = encoder(x)  # [4, 128]
"""

import os
import yaml
import torch

# 이 디렉터리에서 spectral_foundation_v2.py를 임포트할 수 있도록 허용
_DIR = os.path.dirname(os.path.abspath(__file__))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "spectral_foundation_v2",
    os.path.join(_DIR, "spectral_foundation_v2.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SpectralEncoderV2 = _mod.SpectralEncoderV2

_DEFAULT_CKPT = os.path.join(_DIR, "checkpoint_best.pth")
_DEFAULT_CFG = os.path.join(_DIR, "config.yaml")


def _load_checkpoint(checkpoint_path=None, device="cpu"):
    """체크포인트 딕셔너리와 설정을 로드합니다."""
    checkpoint_path = checkpoint_path or _DEFAULT_CKPT
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # yaml 파일 또는 체크포인트에서 설정 로드
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        raise FileNotFoundError(
            f"체크포인트 옆에 config.yaml이 없고 체크포인트 딕셔너리에도 설정이 없습니다"
        )

    # 모델 state_dict 추출
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    return state_dict, config


def _extract_encoder_state_dict(state_dict):
    """전체 MAE state_dict에서 인코더 가중치를 추출하고 'encoder.' 접두사를 제거합니다."""
    encoder_sd = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            new_key = key[len("encoder."):]
            # 데이터셋별 파장 텐서 건너뛰기 (추론에 불필요)
            if new_key.startswith("_wl_tensor_"):
                continue
            encoder_sd[new_key] = value
    return encoder_sd


def load_encoder(
    checkpoint_path=None,
    band_dim=None,
    wavelengths=None,
    device="cpu",
):
    """
    사전학습된 SpectralEncoderV2를 로드합니다.

    Args:
        checkpoint_path: 체크포인트 .pth 파일 경로 (기본값: 이 디렉터리의 checkpoint_best.pth)
        band_dim: 스펙트럼 밴드 수 재설정. None이면 체크포인트의
                  target_bands를 사용합니다 (v71의 경우 80). 다운스트림 태스크에서는
                  사용자 데이터셋의 밴드 수로 설정하세요.
        wavelengths: 위치 인코딩을 위한 파장 값 텐서 (nm).
                     None이면 400-2500 nm 선형 보간을 사용합니다.
                     최적 결과를 위해 실제 데이터셋의 파장을 전달하세요.
        device: 로드할 디바이스 ('cpu', 'cuda', 'cuda:0' 등)

    Returns:
        eval 모드의 SpectralEncoderV2
    """
    state_dict, config = _load_checkpoint(checkpoint_path, device)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    embed_dim = model_cfg.get("embed_dim", 128)
    depth = model_cfg.get("depth", 8)
    num_heads = model_cfg.get("num_heads", 16)
    mlp_ratio = model_cfg.get("mlp_ratio", 4.0)
    if band_dim is None:
        band_dim = data_cfg.get("target_bands", 80)

    # 제공된 파장 사용 또는 위치 인코딩을 위한 플레이스홀더 생성
    if wavelengths is None:
        wavelengths = torch.linspace(400, 2500, band_dim)

    encoder = SpectralEncoderV2(
        embed_dim=embed_dim,
        depth=depth,
        heads=num_heads,
        wavelengths=wavelengths,
        dim_feedforward=int(embed_dim * mlp_ratio),
        band_dim=band_dim,
        activation='relu',
    )

    encoder_sd = _extract_encoder_state_dict(state_dict)
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if missing:
        print(f"[load_encoder] 누락된 키 (band_dim이 다르면 정상): {missing}")
    if unexpected:
        print(f"[load_encoder] 예상치 못한 키 (무시됨): {unexpected}")

    encoder.to(device).eval()
    return encoder


if __name__ == "__main__":
    print("v71 인코더 로딩 중...")
    encoder = load_encoder()
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"인코더 로드 완료: {total_params:,} 파라미터")

    x = torch.randn(4, 80)
    with torch.no_grad():
        out = encoder(x)
    print(f"입력:  {x.shape}")
    print(f"출력: {out.shape}")
    assert out.shape == (4, 128), f"(4, 128)을 기대했으나 {out.shape}을 받았습니다"
    print("검증 통과!")
