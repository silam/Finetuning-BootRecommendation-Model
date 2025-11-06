# 🥾 Boot Recommendation LLM Fine-Tuning with LoRA

This project demonstrates how to **fine-tune a Large Language Model (LLM)** (based on Facebook’s OPT) using **LoRA (Low-Rank Adaptation)** for a custom **boot recommendation assistant**.  
It uses the Hugging Face Transformers, PEFT, and Datasets libraries, with data stored in JSON format.

---

## 🚀 Overview

This notebook fine-tunes the [`facebook/opt-1.3b`](https://huggingface.co/facebook/opt-1.3b) model on a dataset of instructions and responses related to **boot and footwear recommendations**.  
LoRA allows fine-tuning large models efficiently on limited hardware such as Google Colab.

The process includes:
1. Installing dependencies  
2. Loading and preparing dataset  
3. Tokenizing and formatting data  
4. Applying LoRA fine-tuning  
5. Evaluating and saving the trained model  
6. Testing model predictions  
7. Uploading to Hugging Face Hub  

---

## 📦 Dependencies

Install all required Python libraries:
```bash
pip install -q transformers peft accelerate datasets bitsandbytes
```

---

## 📚 Project Structure

```
├── boot_recommendations.json       # Dataset (instruction + response pairs)
├── lora-boot-recommendation.ipynb  # Colab notebook or Python script
├── results/                        # Fine-tuning output directory
├── lora-boot-recommendation-finetuned-model/ # Saved model checkpoint
└── README.md                       # This file
```

---

## 🧠 Model Fine-Tuning Steps

### 1️⃣ Load Dataset
The dataset should be a JSON file with fields:
```json
{
  "instruction": "What boot is good for hiking?",
  "response": "I recommend waterproof boots with ankle support, such as Timberland or Merrell."
}
```

Load it:
```python
from datasets import load_dataset
dataset = load_dataset("json", data_files="/content/drive/MyDrive/Colab Notebooks/boot_recommendations.json")
dataset = dataset["train"].train_test_split(test_size=0.2)
```

---

### 2️⃣ Load and Prepare Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

---

### 3️⃣ Tokenize the Data
```python
def format_example(example):
    return f"### Instruction:\n{example['instruction']}\n### Response:\n{example['response']}"

def tokenize_function(example):
    text = format_example(example)
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
```

---

### 4️⃣ Apply LoRA Configuration
```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

Attach LoRA adapters:
```python
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
model = get_peft_model(model, lora_config)
```

---

### 5️⃣ Training Configuration
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_total_limit=1,
    eval_strategy="steps",
    fp16=True,
)
```

Train:
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
trainer.train()
```

---

### 6️⃣ Save Fine-Tuned Model
```python
trainer.save_model("./lora-boot-recommendation-finetuned-model")
tokenizer.save_pretrained("./lora-boot-recommendation-finetuned-model")
```

---

### 7️⃣ Test the Model
```python
from transformers import pipeline

pipe = pipeline("text-generation",
                model="./lora-boot-recommendation-finetuned-model",
                tokenizer=tokenizer,
                max_new_tokens=100)

prompt = "### Instruction:\nWhat boot do you recommend for winter?\n### Response:\n"
result = pipe(prompt)
print(result[0]['generated_text'])
```

---

### 8️⃣ Upload to Hugging Face Hub
```python
from huggingface_hub import login
from google.colab import userdata

hf_token = userdata.get("HF_TOKEN")
login(hf_token)

model.push_to_hub("sichilam/boot-recommendation-finetuned-model")
tokenizer.push_to_hub("sichilam/boot-recommendation-finetuned-model")
```

---

## 🧩 Notes

- For CPU-only environments, replace `facebook/opt-1.3b` with `facebook/opt-350m`.  
- The dataset can be easily extended for other product categories (e.g., jackets, shoes, accessories).  
- LoRA adapters make the model lightweight and efficient, requiring far less GPU memory than full fine-tuning.  

---

## 💡 Example Output

**Input Prompt:**
```
### Instruction:
What boot do you recommend for hiking in snow?
### Response:
```

**Generated Output:**
```
I recommend insulated, waterproof hiking boots with strong grip, such as Columbia Bugaboot or Salomon Quest Winter boots.
```

---

## 📤 Hugging Face Model Card
Once uploaded, you can find your model here:
👉 [https://huggingface.co/sichilam/boot-recommendation-finetuned-model](https://huggingface.co/sichilam/boot-recommendation-finetuned-model)

---

## 🧾 License
This project is released under the **MIT License**.  
Feel free to modify, retrain, or extend the model for your own applications.
