import torch
from ProtonXOCR import ProtonXOCRConfig, ProtonXOCRModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = ProtonXOCRConfig(shape_probe=True)  # dryrun: dùng stub, không tải SigLIP/Qwen
model = ProtonXOCRModel(cfg).eval().to(device)
print(model)

B, H, W = 1, 384, 384
images = torch.randn(B, 3, H, W, device=device)
T = 16
input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)

with torch.no_grad():
    out = model(images=images, input_ids=input_ids)

for k in ["local_map","local_tokens","global_tokens","global_tokens_grid","visual_embeds","inputs_embeds","logits"]:
    print(k, tuple(out[k].shape))