from transformers import AutoTokenizer, MBartForConditionalGeneration

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-en-ro")

example_english_phrases = ["42 is the answer", "What is the meaning of life?"]
inputs = tokenizer(example_english_phrases, padding=True, max_length=1024, return_tensors="pt")

# Translate
generated_ids = model.generate(**inputs, num_beams=1, max_length=10)
output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("------" * 8)
print(output)
print("------" * 8)
# 42 este răspunsul
