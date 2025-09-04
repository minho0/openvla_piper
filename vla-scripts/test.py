#!/usr/bin/env python3
# test.py — OpenVLA local inference with optional terminal prompt input

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

try:
    from transformers import BitsAndBytesConfig  # only for 4/8-bit
    HAS_BNB = True
except Exception:
    HAS_BNB = False

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def get_openvla_prompt(instruction: str, model_path: str) -> str:
    if "v01" in model_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

def load_image(image_path: str = None, image_url: str = None) -> Image.Image:
    if image_path:
        return Image.open(image_path).convert("RGB")
    if image_url:
        import requests
        return Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    raise ValueError("Provide either --image_path or --image_url.")

def pick_attn_impl(prefer_flash: bool) -> str:
    if not prefer_flash:
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        print("[!] flash-attn not available; falling back to sdpa.")
        return "sdpa"

def build_model_and_processor(model_path: str, mode: str, device: str, prefer_flash: bool):
    attn_impl = pick_attn_impl(prefer_flash)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    kwargs = dict(
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if mode == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
        vla = AutoModelForVision2Seq.from_pretrained(model_path, **kwargs).to(device)
        input_dtype = torch.bfloat16

    elif mode in ("8bit", "4bit"):
        if not HAS_BNB:
            raise RuntimeError("bitsandbytes not found. Install it or use --mode bf16.")
        kwargs["torch_dtype"] = torch.float16
        qcfg = BitsAndBytesConfig(load_in_8bit=True) if mode == "8bit" else BitsAndBytesConfig(load_in_4bit=True)
        kwargs["quantization_config"] = qcfg
        vla = AutoModelForVision2Seq.from_pretrained(model_path, **kwargs)  # bnb가 내부적으로 장치 핸들
        input_dtype = torch.float16
    else:
        raise ValueError("mode must be one of: bf16, 8bit, 4bit")

    return processor, vla, input_dtype

def main():
    parser = argparse.ArgumentParser(description="OpenVLA local inference (single image)")
    parser.add_argument("--model_path", type=str, default="openvla/openvla-7b")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Instruction for the robot (if omitted, will prompt in terminal)")
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig")
    parser.add_argument("--mode", type=str, choices=["bf16", "8bit", "4bit"], default="4bit")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--image_url", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prefer_flash_attn", action="store_true")
    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--print_prompt", action="store_true", help="Print the final text prompt")
    args = parser.parse_args()

    # Ask in terminal if instruction omitted
    if args.instruction is None:
        try:
            print("\n===============================")
            print("🌍  Hello, world! ")
            print("🤖  Please tell your instruction")
            print("===============================\n")
            args.instruction = input("👉 Your command: ").strip()
            print(f"\n✅ Instruction received: {args.instruction}\n")
        except EOFError:
            raise SystemExit("No instruction provided. Exiting...")


    print(f"[*] device: {args.device}")
    print(f"[*] mode: {args.mode}  (bf16 needs more VRAM; 4/8bit needs bitsandbytes)")
    print(f"[*] model: {args.model_path}")
    print(f"[*] instruction: {args.instruction}")

    # Load image
    image = load_image(args.image_path, args.image_url)

    # Build model/processor
    processor, vla, input_dtype = build_model_and_processor(
        args.model_path, args.mode, args.device, args.prefer_flash_attn
    )

    # Prepare prompt & inputs
    prompt = get_openvla_prompt(args.instruction, args.model_path)
    if args.print_prompt:
        print(f"\n[Prompt]\n{prompt}\n")
    target_device = args.device if args.mode == "bf16" else ("cuda" if torch.cuda.is_available() else "cpu")
    inputs = processor(prompt, image).to(target_device, dtype=input_dtype)

    # Inference
    t0 = time.time()
    action = vla.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False) # 일단 bridge_orig 기반으로 역정규화 했는데 , .. ㅇㅋ
    dt = time.time() - t0

    print(f"[OK] Inference time: {dt:.3f}s")
    print("Action [x, y, z, roll, pitch, yaw, gripper]:")
    print(action)

    # Optional: save to JSON
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"action": action.tolist() if hasattr(action, "tolist") else action}, f, indent=2)
        print(f"[*] Saved to {args.save_json}")

if __name__ == "__main__":
    main()