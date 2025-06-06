from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
class ResponseGenerator:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="cpu",
            quantization_config=quant_config,
            resume_download=True
        )
        self.model.eval()
    
    def clean_citations_and_questions(self, text):
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

        # Remove entire sentences or lines that are questions
        text = re.sub(r'^.*\?.*$', '', text, flags=re.MULTILINE)       # Full question lines
        text = re.sub(r'[^.?!]*\?[^.?!]*[.?!]', '', text)              # Embedded questions
        
        # Clean extra spacing artifacts
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def generate_response(self, query, context):
        escaped = context.strip()
        prompt = f"""Use only the PASSAGE below to answer the QUESTION. Just give a factual, concise, and medically accurate answer based strictly on the PASSAGE.
QUESTION: {query}
PASSAGE: {escaped}

ANSWER:"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model.generate(
                **inputs, max_new_tokens=256,do_sample=False,repetition_penalty=1.15
            )
            full_output  = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the decoded output (optional)
            answer = full_output.split("ANSWER:")[-1].strip()
            answer = self.clean_citations_and_questions(answer)
            return answer
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, something went wrong."
