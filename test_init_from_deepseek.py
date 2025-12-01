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

	def _vec_stats(name, v):
		v1 = v.detach().float().cpu().view(-1)
		h = ", ".join([f"{x:.4f}" for x in v1[:8].tolist()])
		print(f"{name}: shape={tuple(v.shape)}, head8=[{h}]  | mean={v1.mean().item():.4f}, std={v1.std().item():.4f}, norm={v1.norm().item():.4f}")

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

	# =========== Extra: print representative vectors per block ===========
	if all(k in out for k in ["local_map", "local_tokens", "global_tokens", "global_tokens_grid", "visual_embeds"]):
		local_map = out["local_map"]                # (B, 1024, Hf, Wf)
		local_tokens = out["local_tokens"]          # (B, N, 1024)
		global_tokens = out["global_tokens"]        # (B, 1+N, 1024)
		global_tokens_grid = out["global_tokens_grid"]  # (B, N, 1024)
		visual_embeds = out["visual_embeds"]        # (B, N, Dh)

		B, C3, Hf, Wf = local_map.shape
		assert B == 1, "Demo assumes batch size 1"
		center_h, center_w = Hf // 2, Wf // 2
		center_idx = center_h * Wf + center_w

		# Local map center vector (C=1024)
		v_local_map_center = local_map[0, :, center_h, center_w]
		_vec_stats("vec.local_map(center_hw)", v_local_map_center)

		# Local token at the same spatial position (C=1024)
		v_local_tok_center = local_tokens[0, center_idx, :]
		_vec_stats("vec.local_tokens(center_idx)", v_local_tok_center)

		# Global CLS and center patch token (C=1024)
		v_global_cls = global_tokens[0, 0, :]
		_vec_stats("vec.global_tokens(CLS)", v_global_cls)
		v_global_patch_center = global_tokens_grid[0, center_idx, :]
		_vec_stats("vec.global_tokens_grid(center_idx)", v_global_patch_center)

		# Fused token (2048) reconstructed from global/local tokens
		import torch as _torch
		v_fused_center = _torch.cat([v_global_patch_center, v_local_tok_center], dim=-1)
		_vec_stats("vec.fused(center_idx)", v_fused_center)

		# Projected token (Dh = 1024)
		v_projected_center = visual_embeds[0, center_idx, :]
		_vec_stats("vec.visual_embeds(center_idx)", v_projected_center)

		# Optional: text embedding first token (shape-probe path only)
		try:
			text_emb = model.text_embedding(input_ids)[0, 0, :]
			_vec_stats("vec.text_embeds(token0)", text_emb)
		except Exception:
			pass

	print("[done] Test finished successfully.")


if __name__ == "__main__":
	main()


