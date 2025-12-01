import torch
from ProtonXOCR import ProtonXOCRConfig, ProtonXOCRModel


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Enable real Qwen3-0.6B
	cfg = ProtonXOCRConfig(
		shape_probe=False,
	)
	model = ProtonXOCRModel(cfg).eval().to(device)

	# Prepare dummy inputs
	B, H, W = 1, 384, 384
	images = torch.randn(B, 3, H, W, device=device)

	# Use real tokenizer if available; fallback to random ids
	if getattr(model, "tokenizer", None) is not None:
		text = "ProtonX OCR quick check."
		enc = model.tokenizer(
			text,
			return_tensors="pt",
			add_special_tokens=True,
			padding=False,
			truncation=True,
		)
		input_ids = enc["input_ids"].to(device)
	else:
		T = 16
		input_ids = torch.randint(0, 151936, (B, T), device=device)

	with torch.no_grad():
		out = model(images=images, input_ids=input_ids)

	# Print a concise summary
	if hasattr(out, "logits"):
		print("llm_output.logits:", tuple(out.logits.shape))
	else:
		try:
			# Some HF models may return dict-like
			print("llm_output.keys:", list(out.keys()))
		except Exception:
			print("llm_output.type:", type(out))


if __name__ == "__main__":
	main()


