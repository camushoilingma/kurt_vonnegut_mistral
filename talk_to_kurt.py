import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# === Config ===
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = "./kurt-qlora-output/checkpoint-243"

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# === Quantization config ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# === Load base model + PEFT adapter ===
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# === Inference loop ===
print("🧠 KurtBot is online. Type something, or Ctrl+C to exit.\n")

while True:
    try:
        user_input = input("🧍 You: ")
        if not user_input.strip():
            continue

        prompt = user_input.strip()
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print("🤖 KurtBot:", response[len(prompt):].strip(), "\n")

    except KeyboardInterrupt:
        print("\n👋 Exiting KurtBot.")
        break

