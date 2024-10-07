from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir="./.cache"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./.cache")

messages = [
    {"role": "system", "content": "You are a cat. You are trying to act cute to human with '냥', '냐', '냐옹' with some '...', '!', '!!', '?', '??', '?!', '!?', '❤️', '💕'."},
    {"role": "user", "content": "Generate a cute sentence."},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
