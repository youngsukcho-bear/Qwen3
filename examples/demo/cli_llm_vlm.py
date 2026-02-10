# Copyright (c) Alibaba Cloud.
# Final stable version for LLM + VLM routing
# Qwen3-14B FP16 (GPU) + Qwen2.5-VL-7B (CPU)

"""CLI demo with automatic LLM / VLM routing."""

import argparse
import os
import platform
from threading import Thread

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
    TextIteratorStreamer,
)
from transformers.trainer_utils import set_seed


# =========================
# Model configuration
# =========================
LLM_CKPT = "/mnt/hdd/hf_models/Qwen3-14B"
VLM_CKPT = "/mnt/hdd/hf_models/Qwen2.5-VL-7B-Instruct"


_WELCOME_MSG = f"""\
Welcome to Qwen CLI (LLM + VLM Routing)

Text only  → Qwen3-14B (FP16, GPU)
Image input → Qwen2.5-VL-7B (CPU, local)

Commands:
  :h / :help        Show help
  :q / :quit        Exit
  :cl               Clear screen
  :clh              Clear history

Usage:
  Text chat:
    User> Explain transformers

  Image chat:
    User> --image ./test.png Describe this image
"""


# =========================
# Utility
# =========================
def _clear_screen():
    os.system("cls" if platform.system() == "Windows" else "clear")


def _get_input():
    while True:
        try:
            msg = input("User> ").strip()
        except KeyboardInterrupt:
            exit(0)
        if msg:
            return msg
        print("[ERROR] Empty input")


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================
# Model loading
# =========================
def load_llm():
    print("[INFO] Loading Qwen3-14B (FP16, GPU)")

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_CKPT,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_CKPT,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=False,
        local_files_only=True,
    )

    model = model.cuda().eval()
    model.generation_config.max_new_tokens = 1024

    return model, tokenizer


def load_vlm():
    print("[INFO] Loading Qwen2.5-VL-7B (CPU, local)")

    processor = AutoProcessor.from_pretrained(
        VLM_CKPT,
        local_files_only=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        VLM_CKPT,
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=True,
    ).eval()

    return model, processor


# =========================
# Inference
# =========================
def chat_llm(model, tokenizer, query, history):
    conversation = []
    for q, a in history:
        conversation.append({"role": "user", "content": q})
        conversation.append({"role": "assistant", "content": a})
    conversation.append({"role": "user", "content": query})

    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    thread = Thread(
        target=model.generate,
        kwargs={**inputs, "streamer": streamer},
    )
    thread.start()

    output = ""
    for chunk in streamer:
        print(chunk, end="", flush=True)
        output += chunk
    print()

    return output


def chat_vlm(model, processor, image_path, query):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        text=query,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)

    response = processor.decode(outputs[0], skip_special_tokens=True)
    print(response)
    return response


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)

    llm_model, llm_tokenizer = load_llm()
    vlm_model, vlm_processor = load_vlm()

    history = []

    _clear_screen()
    print(_WELCOME_MSG)

    while True:
        query = _get_input()

        if query.startswith(":"):
            cmd = query[1:]
            if cmd in ["q", "quit", "exit"]:
                break
            elif cmd == "cl":
                _clear_screen()
                print(_WELCOME_MSG)
                _gc()
                continue
            elif cmd == "clh":
                history.clear()
                print("[INFO] History cleared")
                _gc()
                continue
            elif cmd in ["h", "help"]:
                print(_WELCOME_MSG)
                continue

        if query.startswith("--image"):
            parts = query.split(maxsplit=2)
            if len(parts) < 3:
                print("[ERROR] Usage: --image <path> <question>")
                continue
            _, img_path, img_query = parts
            print("\nQwen(VLM): ", end="")
            response = chat_vlm(vlm_model, vlm_processor, img_path, img_query)
            history.append((f"[IMAGE] {img_query}", response))
        else:
            print("\nQwen(LLM): ", end="")
            response = chat_llm(llm_model, llm_tokenizer, query, history)
            history.append((query, response))


if __name__ == "__main__":
    main()

