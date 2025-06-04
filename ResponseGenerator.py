from transformers import AutoTokenizer, AutoModelForCausalLM
class ResponseGenerator:
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,resume_download=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype="auto",device_map="auto",resume_download=True)
        self.model.eval()

    def generate_response(self, query, context):
        escaped = context.strip()
        prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below.
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
Answer each question clearly and concisely based only on the passage."
QUESTION: {query}
PASSAGE: {escaped}

ANSWER:"""     
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs
            )
            full_output  = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the decoded output (optional)
            answer = full_output.split("ANSWER:")[-1].strip()
            return answer
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, something went wrong."
