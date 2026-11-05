"""Ask a question to a trained checkpoint: python scripts/chat.py runs/<run>/checkpoints/step-<N> "question"."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trener.architectures import read_architecture  # noqa: E402
from trener.chat_format import load_tokenizer  # noqa: E402
from trener.modeling import build_policy  # noqa: E402
from trener.prompts import student_system_prompt  # noqa: E402
from trener.trainer import load_model_weights  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="checkpoint directory (model.safetensors or a LoRA adapter)")
    parser.add_argument("question", help="question text (for choice, the options go on separate lines)")
    parser.add_argument("--base", default="models/LFM3-1.7B", help="source model directory (tokenizer and code)")
    parser.add_argument("--answer-type", default="text", choices=["number", "text", "choice"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--greedy", action="store_true", help="no sampling")
    args = parser.parse_args()

    # the end of an assistant turn is EOS (MiniCPM5: `<|im_end|>`, not `</s>`)
    tokenizer = load_tokenizer(args.base, trust_remote_code=True)
    # the model is built as its architecture prescribes (`architectures` of the model's config.json)
    model = read_architecture(args.base).load(
        Path(args.base), dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
    )
    # a LoRA checkpoint without model.safetensors is rebuilt from its base and the adapter
    policy = build_policy(model, tokenizer, model_path=Path(args.base), device=torch.device("cuda"), with_master=False)
    load_model_weights(policy, args.checkpoint)
    model.eval()

    messages = [
        {"role": "system", "content": student_system_prompt(args.answer_type)},
        {"role": "user", "content": args.question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            temperature=None if args.greedy else 1.0,
            top_p=None if args.greedy else 1.0,
        )
    print(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
