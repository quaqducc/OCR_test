import torch
from typing import Dict, Tuple, Callable


def _load_raw_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
	"""
	Load a checkpoint file and return a plain state_dict-like mapping.
	- Accepts files saved as full dicts or wrapped under common keys.
	"""
	state = torch.load(checkpoint_path, map_location="cpu")
	if isinstance(state, dict):
		for key in ["state_dict", "model", "module", "model_state_dict"]:
			if key in state and isinstance(state[key], dict):
				return state[key]
		return state
	return state


def _copy_matching(
	target_module: torch.nn.Module,
	src_state: Dict[str, torch.Tensor],
	name_map_fn: Callable[[str], str],
) -> Tuple[int, int]:
	"""
	Copy parameters/buffers from src_state into target_module using a name mapper.
	- Only copies when both key exists and tensor shapes match.
	- Returns (num_copied, total_params).
	"""
	target_sd = target_module.state_dict()
	new_sd: Dict[str, torch.Tensor] = {}
	copied, total = 0, 0
	for tkey, tval in target_sd.items():
		total += 1

		# 1) Try mapped key directly
		skey = name_map_fn(tkey)
		if skey in src_state and src_state[skey].shape == tval.shape:
			new_sd[tkey] = src_state[skey]
			copied += 1
			continue

		# 2) Fallback: try suffix match (last resort, but shape must match)
		matches = [k for k, v in src_state.items() if k.endswith(tkey) and v.shape == tval.shape]
		if matches:
			new_sd[tkey] = src_state[matches[0]]
			copied += 1
		else:
			# Keep existing value if we cannot match safely
			new_sd[tkey] = tval

	# Load with strict=False so unmatched keys are ignored gracefully
	target_module.load_state_dict(new_sd, strict=False)
	return copied, total


def init_local_encoder_from_deepseek(local_encoder: torch.nn.Module, checkpoint_path: str) -> Tuple[int, int]:
	"""
	Initialize ProtonX ImageEncoderViT from DeepSeekOCR checkpoints.
	- Accepts checkpoints where local encoder lives under:
	  'image_encoder.*', 'sam_model.*', or 'vision_tower_high.image_encoder.*' (and variants).
	"""
	src = _load_raw_state(checkpoint_path)

	def map_local_name(tkey: str) -> str:
		# Try exact match first
		if tkey in src:
			return tkey
		# Try common prefixes used by DeepSeekOCR
		for prefix in [
			# direct/local
			"image_encoder.",
			"sam_model.",
			# wrapped under 'model.'
			"model.image_encoder.",
			"model.sam_model.",
			# vision tower nesting
			"vision_tower_high.image_encoder.",
			"model.vision_tower_high.image_encoder.",
			"vision_tower_high.",
			"model.vision_tower_high.",
		]:
			cand = prefix + tkey
			if cand in src:
				return cand
		# Return original to allow suffix fallback
		return tkey

	return _copy_matching(local_encoder, src, map_local_name)


def init_clip_from_deepseek(clip_model: torch.nn.Module, checkpoint_path: str) -> Tuple[int, int]:
	"""
	Initialize ProtonX CLIP-like ViT (ClipVitModel) from DeepSeekOCR VitModel.
	- Handles common naming differences:
	  pre_layrnorm ↔ pre_ln, layer_norm{1,2} ↔ ln{1,2},
	  transformer.layers.* ↔ transformer.* (ModuleList index),
	  embeddings/attention/mlp names are preserved where possible.
	"""
	src = _load_raw_state(checkpoint_path)

	def map_clip_name(tkey: str) -> str:
		skey = tkey

		# pre-layernorm rename
		skey = skey.replace("pre_ln.", "pre_layrnorm.")

		# block layernorm rename
		skey = skey.replace(".ln1.", ".layer_norm1.")
		skey = skey.replace(".ln2.", ".layer_norm2.")

		# ModuleList name difference
		# ProtonX: transformer.{i}.*
		# DeepSeek: transformer.layers.{i}.*
		if skey.startswith("transformer.") and not skey.startswith("transformer.layers."):
			skey = skey.replace("transformer.", "transformer.layers.", 1)

		# Most other submodule names (qkv_proj, out_proj, mlp.fc1/fc2, embeddings.*) align already

		# Some checkpoints put VitModel under 'vision_model.' or 'model.vision_model.'
		for prefix in ["vision_model.", "model.vision_model.", ""]:
			full_key = prefix + skey
			if full_key in src:
				return full_key

		# Return mapped key to allow suffix fallback
		return skey

	return _copy_matching(clip_model, src, map_clip_name)


def init_from_deepseek(
	model: torch.nn.Module,
	local_ckpt: str = "",
	clip_ckpt: str = "",
) -> Dict[str, Tuple[int, int]]:
	"""
	High-level helper to initialize ProtonXOCR encoders from DeepSeekOCR checkpoints.
	- Pass file paths for local and/or clip checkpoints.
	- Returns copy stats per sub-module.
	"""
	stats: Dict[str, Tuple[int, int]] = {}
	if local_ckpt:
		copied, total = init_local_encoder_from_deepseek(model.local_encoder, local_ckpt)
		stats["local_encoder"] = (copied, total)
	if clip_ckpt:
		copied, total = init_clip_from_deepseek(model.clip, clip_ckpt)
		stats["clip"] = (copied, total)
	return stats


