import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.config_loader import load_config
from src.models.model import AgenticRAGModel
from src.data.prompt import build_system_tools

config = load_config()

model_config = config.get("models")
generation_config = model_config.get("generation")
base_model_path = generation_config.get("local_path")
device = generation_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from: {base_model_path} on {device}...")

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
    attn_implementation="eager"
)

if device != "cuda" and base_model.device.type != device:
    base_model.to(device)

print("Model loaded successfully!")
print(type(base_model))

model = AgenticRAGModel(base_model, tokenizer)

query = "where is the capital of France?"
system_prompt = build_system_tools()
prompt = {"text": f"{system_prompt} {query}"}

inputs = tokenizer(prompt["text"], return_tensors="pt").to(device)
inputs_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

output_ids = model.generate(
        input_ids=inputs_ids,
        attention_mask=attention_mask,
        max_new_tokens=1000,
        max_length_for_gather=10000,
        do_sample=False,
        temperature=0.8,
)

output_ids = output_ids[0][len(inputs_ids[0]) :]
outputs = tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)

print(outputs)