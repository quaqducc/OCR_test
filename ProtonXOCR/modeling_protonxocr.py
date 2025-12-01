import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .config import ProtonXOCRConfig
from .protonx_encoder import ImageEncoderViT, build_clip_l

try:
	from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
	AutoModelForCausalLM = None
	AutoTokenizer = None


class MlpProjector(nn.Module):
	def __init__(self, in_features: int, out_features: int, bias: bool = True):
		super().__init__()
		self.layers = nn.Linear(in_features, out_features, bias=bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layers(x)


class ProtonXOCRModel(nn.Module):
	def __init__(self, cfg: ProtonXOCRConfig):
		super().__init__()
		self.cfg = cfg

		# Local encoder (SAM/ViTDet-like)
		self.local_encoder = ImageEncoderViT(
			img_size=1024,
			patch_size=cfg.patch_size,
			in_chans=3,
			embed_dim=cfg.local_embed_dim,
			depth=cfg.local_depth,
			num_heads=cfg.local_num_heads,
			mlp_ratio=cfg.local_mlp_ratio,
			out_chans=cfg.local_out_chans,
			use_abs_pos=True,
			use_rel_pos=True,
			window_size=14,
			global_attn_indexes=(2, 5, 8, 11) if cfg.local_depth >= 12 else tuple(range(0, max(0, cfg.local_depth - 1), 3)),
		)

		# Global tokens via CLIP-like ViT (replacing SigLIP)
		self.clip = build_clip_l()

		# Decoder (Qwen)
		if cfg.shape_probe or AutoModelForCausalLM is None:
			self.llm = None
			self.tokenizer = None
			self.decoder_hidden_size = cfg.decoder_hidden_size
			self.text_embedding = nn.Embedding(cfg.vocab_size, self.decoder_hidden_size)
			self.lm_head = nn.Linear(self.decoder_hidden_size, cfg.vocab_size, bias=False)
		else:
			self.llm = AutoModelForCausalLM.from_pretrained(cfg.qwen_model_name, trust_remote_code=True)
			self.tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_name, trust_remote_code=True, use_fast=True)
			if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
				self.tokenizer.pad_token = self.tokenizer.eos_token
			self.decoder_hidden_size = int(getattr(self.llm.config, "hidden_size", cfg.decoder_hidden_size))

		# Projector: concat(CLIP 1024, Local 1024) -> LLM hidden size
		self.projector = MlpProjector(
			in_features=self.clip.embeddings.embed_dim + self.local_encoder.net_3.out_channels,
			out_features=self.decoder_hidden_size,
			bias=True,
		)

	def forward(self, images: torch.Tensor, input_ids: Optional[torch.LongTensor] = None):
		device = next(self.parameters()).device
		images = images.to(device)
		if input_ids is not None:
			input_ids = input_ids.to(device)

		# Visual encoder fusion (CLIP + Local)
		local_map = self.local_encoder(images)  # (B, 1024, H', W')
		clip_tokens = self.clip(images, local_map)  # (B, 1+N, 1024)
		clip_patch_tokens = clip_tokens[:, 1:]  # (B, N, 1024)
		B, C3, Hf, Wf = local_map.shape
		local_tokens = local_map.flatten(2).permute(0, 2, 1).contiguous()  # (B, N, 1024)
		fused_tokens = torch.cat([clip_patch_tokens, local_tokens], dim=-1)  # (B, N, 2048)
		visual_embeds = self.projector(fused_tokens)  # (B, N, Dh)

		if self.llm is None:
			assert input_ids is not None, "input_ids required in shape-probe mode"
			text_embeds = self.text_embedding(input_ids)  # (B, T, Dh)
			inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # (B, N+T, Dh)
			logits = self.lm_head(inputs_embeds)  # (B, N+T, vocab)
			return {
				"local_map": local_map,
				"local_tokens": local_tokens,
				"global_tokens": clip_tokens,
				"global_tokens_grid": clip_patch_tokens,
				"clip_tokens": clip_tokens,
				"visual_embeds": visual_embeds,
				"inputs_embeds": inputs_embeds,
				"logits": logits,
			}
		else:
			assert input_ids is not None, "input_ids required"
			text_embeds = self.llm.get_input_embeddings()(input_ids)
			inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
			attn_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)
			return self.llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask)


