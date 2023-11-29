import pandas as pd
from transformers import TFAutoModelForCausalLM, AutoTokenizer
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="pansophic/rocket-3B", trust_remote_code=True)

model = TFAutoModelForCausalLM.from_pretrained("rocket-3B")
tokenizer = AutoTokenizer.from_pretrained("rocket-3B")
def encode_prompt(prompt, tokenizer):
    return tokenizer.encode(prompt, return_tensors='tf', truncation=True, max_length=512)
df = pd.read_csv("LLM-Sample-Input-File.csv")
sample = df.sample(1)
input_text = "Answer: How much revenue does " + sample.iloc[0]['Company'] + " make from selling Smartphones in " + sample.iloc[0]['Market'] + "?"

input_ids = encode_prompt(input_text, tokenizer)
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
actual_revenue = sample.iloc[0]['Revenue']
print("Actual Revenue:", actual_revenue)
print("Model Prediction:", output_text)
#Please note that the provided code only trains the model on a single sample.
