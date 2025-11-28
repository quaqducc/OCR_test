"""
End-to-end test: initialize ProtonXOCR encoders from a single DeepSeek .safetensors
and run a forward pass to verify shapes. Heavily commented for step-by-step clarity.
"""

import os
import sys
import torch


def main():
	# 1) Resolve input path for the DeepSeek .safetensors
	#    - Default to the path you specified; override with CLI arg if needed.
	default_safetensors = r"D:\ProtonX\ProtonX_OCR_A\ProtonXOCR\model-00001-of-000001.safetensors"
	safetensors_path = sys.argv[1] if len(sys.argv) > 1 else default_safetensors
	assert os.path.isfile(safetensors_path), f"Not found: {safetensors_path}"

	# 2) Convert .safetensors -> .pt (once). Our weight_init loader uses torch.load on a single file path.
	#    - We save the entire state dict so both local and clip inits can filter by prefix.
	pt_path = os.path.splitext(safetensors_path)[0] + ".pt"
	if not os.path.isfile(pt_path):
		from safetensors.torch import load_file
		print(f"[convert] Loading safetensors: {safetensors_path}")
		state = load_file(safetensors_path)  # returns a flat state_dict-like mapping
		print(f"[convert] Saving as .pt: {pt_path}")
		torch.save(state, pt_path)
	else:
		print(f"[convert] Reusing existing: {pt_path}")

	# 3) Import ProtonXOCR and the high-level init helper
	from ProtonXOCR import ProtonXOCRConfig, ProtonXOCRModel, init_from_deepseek

	# 4) Create model in shape-probe mode (no actual Qwen download)
	cfg = ProtonXOCRConfig(shape_probe=True)
	model = ProtonXOCRModel(cfg).eval()  # stay on CPU for weight copy

	# 5) Initialize both encoders from the same .pt
	#    - The loader will internally filter by prefixes like:
	#      image_encoder.* / vision_tower_high.image_encoder.*  (local)
	#      vision_model.*                                       (clip)
	print("[init] Loading DeepSeek weights into encoders...")
	stats = init_from_deepseek(
		model,
		local_ckpt=pt_path,
		clip_ckpt=pt_path,
	)
	print(f"[init] Copy stats: {stats}")

	# 6) Move to device and create a tiny test batch
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device).eval()

	B, H, W = 1, 384, 384
	images = torch.randn(B, 3, H, W, device=device)
	T = 16
	input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)

	# 7) Forward pass and print key tensor shapes
	with torch.no_grad():
		out = model(images=images, input_ids=input_ids)

	keys = [
		"local_map",
		"local_tokens",
		"global_tokens",
		"global_tokens_grid",
		"visual_embeds",
		"inputs_embeds",
		"logits",
	]
	for k in keys:
		if k in out:
			print(f"{k}: {tuple(out[k].shape)}")
		else:
			print(f"{k}: (missing)")

	print("[done] Test finished successfully.")


if __name__ == "__main__":
	main()


