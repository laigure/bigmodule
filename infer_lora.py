import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run a quick chat test with a LoRA adapter.")
    parser.add_argument("--base_model", required=True, help="Base model id or local path.")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory.")
    parser.add_argument("--prompt", default="用三句话解释什么是监督微调 SFT。")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16.")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    print(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
