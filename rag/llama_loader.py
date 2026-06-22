# rag/llama_loader.py
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CONFIG_PATH = "artifacts/llama_model/config.pkl"

with open(CONFIG_PATH, "rb") as f:
    cfg = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

model = AutoModelForCausalLM.from_pretrained(
    cfg["model_name"],
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

def generate(prompt: str, max_tokens: int = 300):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
