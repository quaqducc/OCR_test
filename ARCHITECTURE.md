# ProtonXOCR Architecture (Local ViTDet + CLIP-like ViT + Qwen3-0.6B)

## Overview
ProtonXOCR uses a dual-branch visual encoder with per-location fusion:
- Local branch: SAM/ViTDet-like encoder producing high-resolution spatial features (1024 channels)
- Global branch: CLIP-like ViT producing per-patch tokens (1024)
- Decoder: Qwen/Qwen3-0.6B (hidden size 1024)

Tokens from both branches are concatenated per spatial location (2048) and mapped to the LLM hidden size (1024).

## Components
- Local encoder (`ImageEncoderViT` in `ProtonXOCR/protonx_encoder.py`)
  - `PatchEmbed`: Conv2d(3→768, kernel=stride=16)
  - ViT blocks with (LN + MHA + MLP), window/global attention support
  - `neck`: Conv1×1 768→256, LN2d, Conv3×3 256→256, LN2d
  - `net_2`: Conv3×3 stride 2, 256→512
  - `net_3`: Conv3×3 stride 2, 512→1024
  - Output: local feature map `x3` with shape (B, 1024, H/64, W/64) for patch_size=16

- Global encoder (`ClipVitModel` in `ProtonXOCR/protonx_encoder.py`)
  - CLIP-like ViT with learnable class token
  - Patch embedding conv (stride=patch_size), absolute positional embeddings (resized for variable grids)
  - Transformer with MHA + MLP blocks
  - Output: tokens `(B, 1+N, 1024)` where N = (H/patch)·(W/patch); we drop CLS to get `(B, N, 1024)`

- Fusion & Projector
  - Flatten local map to `(B, N, 1024)` where N = H'·W'
  - Align global tokens to N (CLIP grid matches local grid via the shared patch stride path)
  - Concatenate per-token on channel dimension → `(B, N, 2048)`
  - Linear projector: `(2048 → 1024)` to match Qwen3-0.6B hidden size

- Decoder (Qwen3-0.6B)
  - In shape-probe: use `nn.Embedding(vocab, 1024)` and `nn.Linear(1024→vocab)` to avoid downloads
  - In real mode: load `AutoModelForCausalLM`/`AutoTokenizer` for `Qwen/Qwen3-0.6B` and pass `inputs_embeds` = concat(visual_embeds, text_embeds)

## Data Flow
1) Image → Local encoder → local_map: `(B, 1024, H', W')`
2) Image → CLIP-like ViT → tokens `(B, 1+N, 1024)` → drop CLS → `(B, N, 1024)`
3) Local flatten → `(B, N, 1024)`
4) Concat (local, global) → `(B, N, 2048)`
5) Projector → `visual_embeds (B, N, 1024)`
6) Text tokens `(B, T)` → embeddings `(B, T, 1024)`
7) Concat → `inputs_embeds (B, N+T, 1024)`
8) Decoder head → logits (B, N+T, vocab)

## Notes
- Local (ViTDet-like) preserves fine text structure; CLIP-like ViT brings global context.
- Per-location fusion ensures each spatial position carries both local detail and global semantics.
- Defaults: hidden sizes 1024 (both visual branches and Qwen3-0.6B); projector maps 2048→1024.

## Files
- `ProtonXOCR/protonx_encoder.py`: Local encoder and CLIP-like ViT.
- `ProtonXOCR/modeling_protonxocr.py`: Fusion, projector, and Qwen integration.
- `ARCHITECTURE.md`: This document.

## Block Diagram

```mermaid
flowchart TD
    I[Image (B,3,H,W)] --> LE[Local Encoder (ViTDet-like)]
    LE --> NM[Neck: Conv1x1 -> LN2d -> Conv3x3 -> LN2d]
    NM --> C2[net_2: Conv3x3, stride 2 (256->512)]
    C2 --> C3[net_3: Conv3x3, stride 2 (512->1024)]
    C3 --> LF[Flatten to local tokens (B,N,1024)]

    I --> CE[CLIP-like ViT (B,1+N,1024)]
    CE --> DropCLS[Drop CLS] --> GF[Global tokens (B,N,1024)]

    LF --> Cat[Concat per-position]
    GF --> Cat

    Cat --> Proj[Projector Linear (2048 -> 1024)]
    Proj --> VE[Visual embeds (B,N,1024)]

    TE[Text Embedding (B,T,1024)] --> ConcatVT[Concat Visual+Text (B,N+T,1024)]
    VE --> ConcatVT
    ConcatVT --> Dec[Qwen 3-0.6B (CausalLM)] --> Logits[Logits (B,N+T,Vocab)]
```

ASCII fallback:

```
Image (B,3,H,W)
  |\
  | \-- CLIP-like ViT --> tokens (B,1+N,1024) -- drop CLS --> (B,N,1024)
  |
  +--> Local Encoder (ViTDet-like)
          -> Neck (1x1/LN2d -> 3x3/LN2d)
          -> net_2 (3x3 s2, 256->512)
          -> net_3 (3x3 s2, 512->1024)
          -> local_map (B,1024,H',W') -> flatten -> (B,N,1024)

Concat per-position: (B,N,1024) || (B,N,1024) -> (B,N,2048)
Projector Linear 2048 -> 1024 -> visual_embeds (B,N,1024)

Text ids (B,T) -> Embedding (B,T,1024)
Concat visual+text -> (B,N+T,1024)
Qwen 3-0.6B (CausalLM) -> logits (B,N+T,Vocab)
```
