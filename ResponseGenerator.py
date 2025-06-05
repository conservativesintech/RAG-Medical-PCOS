from transformers import AutoTokenizer, AutoModelForCausalLM
import re
class ResponseGenerator:
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,resume_download=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype="auto",device_map="auto",resume_download=True)
        self.model.eval()
    
    def clean_citations(self, text):
        # Remove square bracket numeric citations like [88], [90, 91]
        text = re.sub(r'\[\s?\d+(?:\s?,\s?\d+)*\s?\]', '', text)
        
        # Remove parenthesis numeric citations like (1), (2, 3)
        text = re.sub(r'\(\s?\d+(?:\s?,\s?\d+)*\s?\)', '', text)
        
        # Remove full structured reference-style citations like:
        # "Smith AB. J Clin Med. 2021;20(2):101–105."
        text = re.sub(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\.?\s*)?(?:[A-Za-z\s]+\.)?\s?\d{4};\d+\(.*?\):\d+(–\d+)?\.',
            '', text
        )

        # Clean extra spacing artifacts
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def generate_response(self, query, context):
        escaped = context.strip()
        prompt = f"""You are a concise, scientific assistant. Answer clearly and accurately based on the provided passage..
QUESTION: {query}
PASSAGE: {escaped}

ANSWER:"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model.generate(
                **inputs
            )
            full_output  = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the decoded output (optional)
            answer = full_output.split("ANSWER:")[-1].strip()
            answer = self.clean_citations(answer)
            return answer
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, something went wrong."
