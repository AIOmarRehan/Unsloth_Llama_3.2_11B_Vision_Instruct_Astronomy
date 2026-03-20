[Medium Article](https://medium.com/@ai.omar.rehan/fine-tuning-llama-3-2-11b-vision-on-an-astronomy-dataset-with-unsloth-bb184801564d)

---

[Hugging Face Space](https://huggingface.co/spaces/AIOmarRehan/AstroVision-Llama-3.2-11B)

---


# Astronomy Vision Fine-Tuning with Llama 3.2-11B & Unsloth

This project fine-tunes a vision-language model for astronomy image description using Unsloth and Llama 3.2 11B Vision. The workflow covers dataset inspection, cleaning, multimodal formatting, LoRA fine-tuning, evaluation, deployment, and Hugging Face upload.

## Project Files

- `Notebook/unsloth_Llama_3_2_11B_Vision_Instruct_Astronomy.ipynb`: main notebook for EDA, cleaning, training, evaluation, and export.
- `Notebook/unsloth_Llama_3_2_11B_Vision_Instruct_Astronomy_Deploying.ipynb`: deployment notebook for loading the fine-tuned weights and testing with images.
- `astronomy_dataset/`: local dataset with `data.json` and image files.
- `upload_as_hf_image_text_dataset.py`: simple Python uploader for pushing the dataset to Hugging Face with `image` and `text` columns.
- `quick_upload_to_hf.md`: very short guide for dataset upload.

## Dataset Summary

- 250 image-caption pairs.
- Main columns: `image_id`, `text`, `image`.
- Average caption length: about 93.9 characters and 15.2 words.
- Train/validation/test split used in the notebook: 200 / 25 / 25.
- Heuristic label grouping from captions:
  - Earth: 77
  - Mars: 54
  - Mars Rover: 46
  - Milky Way: 45
  - Hubble: 28

## Main Workflow

1. Load the zipped dataset from Google Drive into Colab.
2. Convert `data.json` into a pandas DataFrame.
3. Run EDA on captions and images.
4. Clean captions, remove duplicates, and validate image files.
5. Convert each row into a multimodal conversation format.
6. Fine-tune `unsloth/Llama-3.2-11B-Vision-Instruct` with LoRA.
7. Evaluate generated captions with BLEU and ROUGE.
8. Export the model and prepare it for deployment.
9. Upload the dataset to Hugging Face in viewer-friendly format.

## Important Code Examples

### Load the base vision model

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
```

### Apply LoRA fine-tuning

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)
```

### Convert one dataset row to multimodal chat format

```python
def convert_to_conversation(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are an expert astronomer. Describe accurately what you see in this image."},
                    {"type": "image", "image": os.path.join(base_path, sample["image"])},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ]
    }
```

### Train with TRL + Unsloth collator

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=hf_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=30,
        learning_rate=2e-4,
        optim="adamw_8bit",
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
    ),
)
```

### Deploy the merged model with a single image prompt

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe accurately what you see in this image."},
        ],
    }
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128, temperature=1.2)
```

## Main Plots and Visual Checks

The main notebook includes these important visual checks:

![`Caption Word Count Distribution`: shows most captions clustered around 14 to 17 words.](https://files.catbox.moe/nkda95.png)

![`Top 20 Most Frequent Words in Captions`: highlights dominant astronomy terms such as `mars`, `earth`, `milky`, and `hubble`.](https://files.catbox.moe/let1ll.png)

![`Image Resolution Distribution`: scatter plot and histograms for image width and height.](https://files.catbox.moe/vftgj0.png)

### Random images with their labels.

![`Label / Class Distribution`: caption-derived class balance across Earth, Mars, Hubble, Milky Way, and Mars Rover.](https://files.catbox.moe/z49ttk.png)

## Evaluation Snapshot

- BLEU: 0.0537
- ROUGE-1: 0.2658
- ROUGE-2: 0.0979
- ROUGE-L: 0.2383
- ROUGE-Lsum: 0.2469

These scores were produced from the saved notebook output and should be treated as an early training snapshot, not a final benchmark.

## Training and Deployment Notes

- The notebook uses 4-bit loading for efficiency during fine-tuning.
- Saved training stats show peak reserved memory around 10.049 GB, with about 1.467 GB attributed to training overhead.
- The deployment notebook loads the base quantized model, applies the LoRA adapter, and supports:
  - direct image testing in Colab
  - Gradio interface deployment
  - pushing weights to Hugging Face

## Tools Used in This Project

- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) 
- ![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white) 
- ![Unsloth](https://img.shields.io/badge/Unsloth-000000?style=flat&logo=unsloth&logoColor=white)
- ![LLaMA](https://img.shields.io/badge/LLaMA-FFA724?style=flat&logo=meta&logoColor=white)
- ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
- ![Transformers](https://img.shields.io/badge/Transformers-000000?style=flat&logo=transformers&logoColor=white) 
- ![TRL](https://img.shields.io/badge/TRL-000000?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABl0lEQVQ4jWNgGAUjEawZs76P8P3z58+GFcgKErQY5kYQAEkO8/+/fvfBOSjAmBmChPJrV69eqZHBDhGDoYwMWDtkCAThCDUr0GUk5AGISQu6fXr1+/BgxYmAEiB7RzE6ohWJiYgEw4xGNmCJFAjI0wPGA0SB0iI4j4hBwE8Ss4wA1AsgCwYgUhjE+5cepUwcJ0V+3fvx8uXHz/4VhgWJ+o+3bt27b3z5s1biJPA0igwQxgYiMSI1mQJCCQG4gTgEEBNI7QpGgDyA0RIugoYrybBwcQdRIDDgYGvw/37t2//6hQ4f/+++/nwMRQL0UCgKzVm0+//btw+gxtQL0UGgDxBKEJLQIBykxhYGBAgSg39+/Di4uL3+fPn/89KJEycPHj5Fjt27ZfIYdjY2GClQ8IwYGBgQGCjR48eLCy+SRQkDYC3QEBQVAEmFDxg1apVq9e/e/z58/vz4BQf6A2BiYkJmZGRi7d++zZs2f3779/Dhw9t27Z58+f75+8GDB9+7dq/z58/f3Hjx2jRq1WvXnz59Onj26NHjx49Yt29fvnz586ePfuXwYGCAo0aNXqFnfPn36cGKlvYOTk5ehYsXqsnfv3+/fv38zZs3b1atXrx8+fPn1q5d+/fw4YNmzZ89evX6/z5s2b/+/fv0qVLl8uXLy8vLaNGjYuXKlSuHfv3smTJ6dOnT+//79+/wIGBC0bNkyVOnDhg3Xrly5d+/e/Zs6c+bN+/fq3bt3at27d+/evXr1///wMDIyCAYHKxYsWKbNm2KFcQYGBh4ePsdOHChV+/fvf/+/fvpcvXrx48aN++/fvz/+/fvwpUtW7b8+fPb744w9+e/fu3YoUPH66+/fuu+++O9u2bXv372/fv379+/fvx48fLkyZOsGDB/fee+/fv36lTp9+/f/0bxA88/Pz92/fv9+/fz8/Pv3799+vXr1//79+/v58+fU1NTX379u3T6tWr27d+9++67d+7cOGDx9+vQwMDIyMjLjy8vL+/fv3//v27dupY8eOxY8eOjo6Pnz59e7cuX766+/Pnz+/fv3799+/bty4ePHl28efPGmzZs2vXr2+/fv+/fv3///v27dvPnz79+/fv3///8TFxQX17969u3b9+/fv379/Pnz9+/fTq1evbtm1atXr9+/bNy5c3b9+/f29vbfv3///+2/NyshxsTExGJDw8fHx8fHz8/PycnJyddNChQ/e3t7e/ePHj///v37z5s3Lly5duvXt3bt2/Pnz9+/fv+/fv2///+/fv3//v37+/v79/Pnz9+/fv379+/v79/Pnz6tSpUoUuXLgQEBBjx49mzZkzt27evXr3+EADDEyMjw8fHx8fHx8PDw8PDw8PDy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLx8fPz8/Pv78fHx8PDw8PDw8PDw8PDw8PDy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLw8PDw8PDw8PDw8PDw8PDw8PDw8PDw8PDw8PDw8PDw8PAAA==)
- ![PEFT](https://img.shields.io/badge/PEFT-LoRA-000000?style=flat&logo=peft&logoColor=white)
- ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FF6F00?style=flat&logo=huggingface&logoColor=white)
- ![huggingface_hub](https://img.shields.io/badge/huggingface__hub-FF6F00?style=flat&logo=huggingface&logoColor=white)
- ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
- ![matplotlib](https://img.shields.io/badge/matplotlib-000000?style=flat&logo=matplotlib&logoColor=white)
- ![Pillow](https://img.shields.io/badge/Pillow-000000?style=flat&logo=pillow&logoColor=white) 
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) 
- ![evaluate](https://img.shields.io/badge/evaluate-000000?style=flat&logo=evaluate&logoColor=white) 
- ![rouge_score](https://img.shields.io/badge/ROUGE_Score-000000?style=flat&logo=rouge&logoColor=white)
- ![langdetect](https://img.shields.io/badge/langdetect-000000?style=flat&logo=langdetect&logoColor=white) 
- ![Gradio](https://img.shields.io/badge/Gradio-FF5B00?style=flat&logo=gradio&logoColor=white) 

---

## Results

<video controls>
  <source src="https://files.catbox.moe/g5x3vt.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
