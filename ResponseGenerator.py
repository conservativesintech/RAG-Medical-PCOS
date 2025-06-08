from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import torch
class ResponseGenerator:
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
            trust_remote_code=True,
            resume_download=True
        )
        self.model.eval()
    
    def clean_citations_and_questions(self, text):
        # Remove square bracket numeric citations like [88], [90, 91]
        text = re.sub(r'\[\s?\d+(?:\s?,\s?\d+)*\s?\]', '', text)
        
        # Remove full structured reference-style citations like:
        # "Smith AB. J Clin Med. 2021;20(2):101–105."
        text = re.sub(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\.?\s*)?(?:[A-Za-z\s]+\.)?\s?\d{4};\d+\(.*?\):\d+(–\d+)?\.',
            '', text
        )
        
        text = re.sub(r'\b\w+\s+et al\.,?\s*\d{0,4}', '', text)
        # Clean extra spacing artifacts
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def generate_response(self, query, context):
        escaped = context.strip()
        prompt = f"""You are a medical assistant trained in PCOS. Answer the QUESTION using only the information provided in the CONTEXT below.
Do not use any outside knowledge or make up information. Be accurate, concise, and factual.

QUESTION: {query}
CONTEXT: {escaped}
ANSWER:"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]
            max_total_tokens = getattr(self.model.config, "max_position_embeddings", self.model.config.max_position_embeddings)
            # Dynamically allocate max_new_tokens based on input length
            max_new_tokens = max(32, min(512, max_total_tokens - input_len))
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,do_sample=False,repetition_penalty=1.1
            )
            full_output  = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the decoded output (optional)
            answer = full_output.split("ANSWER:")[-1].strip()
            answer = self.clean_citations_and_questions(answer)
            return answer
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, something went wrong."
