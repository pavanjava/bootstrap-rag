## Image Analysis with Vision LM

A Python tool for image analysis using Qwen2-VL-2B-Instruct model via Llama-Index.

### Setup Project

1. Create `.env` file with your HuggingFace token:

```
HF_TOKEN=your_huggingface_token_here
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the project:

```bash
python main.py
```

### Requirements

```
huggingface_hub
python-dotenv
llama-index
```

### Supported Models
```text
Qwen2 Vision
Florence2
Phi3.5 Vision
PaliGemma
Mllama
```