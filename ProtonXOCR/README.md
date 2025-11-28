# ProtonXOCR (SigLIP Global + SAM/ViTDet Local + Qwen 0.5B)

ProtonXOCR mirrors DeepseekOCR's encoder–conv–decoder pipeline but swaps:
- Global encoder: CLIP → SigLIP Base
- Decoder: Deepseek → Qwen/Qwen2-0.5B-Instruct

Local encoder remains SAM/ViTDet-like (patch_embed → ViT blocks → neck → net_2 → net_3).

## Structure
- `config.py`: `ProtonXOCRConfig` hyperparameters
- `deepencoder.py`: SAM/ViTDet-like local encoder (`ImageEncoderViT`), utilities, layer norms
- `modeling_protonxocr.py`: `ProtonXOCRModel` wiring local encoder + SigLIP + projector + Qwen
- `__init__.py`: exports

## Quick Start (Shape Probe)
The default config runs in shape-probe mode (no downloads). It uses a SigLIP stub and a tiny decoder stub.

```python
import torch
from ProtonXOCR import ProtonXOCRConfig, ProtonXOCRModel

cfg = ProtonXOCRConfig(shape_probe=True)
model = ProtonXOCRModel(cfg).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

B, H, W = 1, 384, 384
images = torch.randn(B, 3, H, W, device=device)
T = 16
input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)

with torch.no_grad():
    out = model(images=images, input_ids=input_ids)

for k in ["local_map","local_tokens","global_tokens","global_tokens_grid","visual_embeds","inputs_embeds","logits"]:
    print(k, tuple(out[k].shape))
```

## Real Models
Set `shape_probe=False` and ensure `transformers` is installed.

```python
cfg = ProtonXOCRConfig(
    shape_probe=False,
    siglip_model_name="google/siglip-base-patch16-384",
    qwen_model_name="Qwen/Qwen2-0.5B-Instruct",
)
model = ProtonXOCRModel(cfg).eval().to(device)
```

The model will:
1) Encode local map via `ImageEncoderViT` → `(B, 1024, H/64, W/64)`
2) Encode SigLIP global tokens → pool to local grid → `(B, N, 768)`
3) Concat with local tokens `(B, N, 1024)` → projector → `(B, N, 896)`
4) Concat with text embeddings → feed Qwen CausalLM

## Notes
- `ImageEncoderViT` assumes square positional maps; arbitrary sizes are handled by interpolating absolute positions.
- For very high resolutions, windowed attention (window_size=14) reduces memory while mixing periodic global attention layers.
- You can swap Qwen size by adjusting `decoder_hidden_size` and model name if you align the projector accordingly.


