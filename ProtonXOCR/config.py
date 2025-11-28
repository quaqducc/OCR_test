from dataclasses import dataclass


@dataclass
class ProtonXOCRConfig:
	# Image / patch
	image_size: int = 384
	patch_size: int = 16

	# Local encoder (SAM/ViTDet-like)
	local_embed_dim: int = 768
	local_depth: int = 12
	local_num_heads: int = 12
	local_mlp_ratio: float = 4.0
	local_out_chans: int = 256

	# Global encoder (SigLIP)
	siglip_model_name: str = "google/siglip-base-patch16-384"
	global_embed_dim: int = 768

	# Decoder (Qwen 0.5B)
	qwen_model_name: str = "Qwen/Qwen3-0.6B"
	decoder_hidden_size: int = 1024
	vocab_size: int = 151936

	# Projector
	projector_bias: bool = True

	# Runtime
	shape_probe: bool = True  # if True, do not download real models


