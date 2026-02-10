# api_server.py
import io
import os
from typing import List, Optional
from threading import Thread

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
    TextIteratorStreamer,
)
from transformers.trainer_utils import set_seed


# =========================
# Environment
# =========================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# =========================
# Model configuration
# =========================
LLM_CKPT = "/mnt/hdd/hf_models/Qwen3-14B"
VLM_CKPT = "/mnt/hdd/hf_models/Qwen2.5-VL-7B-Instruct"


# =========================
# FastAPI
# =========================
app = FastAPI(title="Qwen LLM + VLM API")


# =========================
# Global (lazy loaded)
# =========================
llm_model = None
llm_tokenizer = None
vlm_model = None
vlm_processor = None


# =========================
# Schemas
# =========================
class ChatRequest(BaseModel):
    query: str
    history: Optional[List[List[str]]] = []


class ChatResponse(BaseModel):
    response: str


# =========================
# Model loaders
# =========================
def load_llm():
    print("[INFO] Loading Qwen3-14B (device_map=auto)")

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_CKPT,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_CKPT,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()

    model.generation_config.max_new_tokens = 1024
    return model, tokenizer


def load_vlm():
    print("[INFO] Loading Qwen2.5-VL-7B (CPU, fp32)")

    processor = AutoProcessor.from_pretrained(
        VLM_CKPT,
        local_files_only=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        VLM_CKPT,
        torch_dtype=torch.float32,
        device_map="cpu",
        local_files_only=True,
    ).eval()

    return model, processor


# =========================
# Lazy getters
# =========================
def get_llm():
    global llm_model, llm_tokenizer
    if llm_model is None:
        llm_model, llm_tokenizer = load_llm()
    return llm_model, llm_tokenizer


def get_vlm():
    global vlm_model, vlm_processor
    if vlm_model is None:
        vlm_model, vlm_processor = load_vlm()
    return vlm_model, vlm_processor


# =========================
# Startup
# =========================
@app.on_event("startup")
def startup():
    set_seed(1234)
    print("[INFO] API server started (lazy loading enabled)")


# =========================
# LLM (non-streaming)
# =========================
@app.post("/chat", response_model=ChatResponse)
def chat_llm_api(req: ChatRequest):
    model, tokenizer = get_llm()

    conversation = []
    for q, a in req.history:
        conversation.append({"role": "user", "content": q})
        conversation.append({"role": "assistant", "content": a})
    conversation.append({"role": "user", "content": req.query})

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs)

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True,
    )

    return ChatResponse(response=response)


# =========================
# LLM (streaming)
# =========================
@app.post("/chat/stream")
def chat_llm_stream(req: ChatRequest):
    model, tokenizer = get_llm()

    conversation = []
    for q, a in req.history:
        conversation.append({"role": "user", "content": q})
        conversation.append({"role": "assistant", "content": a})
    conversation.append({"role": "user", "content": req.query})

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    def generate():
        thread = Thread(
            target=model.generate,
            kwargs={
                **inputs,
                "streamer": streamer,
                "max_new_tokens": 1024,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )
        thread.start()

        for chunk in streamer:
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")


# =========================
# VLM (image, non-streaming)
# =========================
@app.post("/chat/image", response_model=ChatResponse)
def chat_vlm_api(
    query: str = Form(...),
    image: UploadFile = File(...),
):
    model, processor = get_vlm()

    image_bytes = image.file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=pil_image,
        text=prompt,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
        )

    response = processor.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return ChatResponse(response=response)


# =========================
# Health check
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}
