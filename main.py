from DocumentRetriever import DocumentRetriever
from ResponseGenerator import ResponseGenerator
from DocumentQA import DocumentQA
import json
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer, util

def compute_f1(reference, prediction):
    ref_tokens = set(word_tokenize(reference.lower()))
    pred_tokens = set(word_tokenize(prediction.lower()))
    common = ref_tokens.intersection(pred_tokens)

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def main():
    print("Building document index...")
    retriever = DocumentRetriever(data_dir='Data')
    generator = ResponseGenerator()
    qa_system = DocumentQA(retriever, generator)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    with open("pcos_qna.json", "r", encoding="utf-8") as file:
        pcos_qna = json.load(file)

    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    total_bleu, total_rougeL, total_f1 = 0, 0, 0
    count = 0
    hallucination_data = []

    for question, reference in pcos_qna.items():
        try:
            # Get generated answer
            prediction = qa_system.get_answer(question)

            # Fallback for empty or unhelpful answers
            if not prediction or prediction.lower().strip() in ["", "i don't know", "not found in the passage."]:
                prediction = "Not found in the passage."

            # Get retrieved chunks
            hallucination_result = qa_system.hallucination_test(question)
            retrieved_chunks = hallucination_result.get("retrieved_chunks", [])

            # BLEU-2 Score
            ref_tokens = word_tokenize(reference.lower())
            pred_tokens = word_tokenize(prediction.lower())
            bleu = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5), smoothing_function=smooth)

            # ROUGE-L Score
            rouge_scores = rouge.score(reference, prediction)
            rouge_l = rouge_scores['rougeL'].fmeasure

            # F1 Score
            f1 = compute_f1(reference, prediction)
            # Semantic similarity for hallucination labeling
            emb_ref = embed_model.encode(reference, convert_to_tensor=True)
            emb_pred = embed_model.encode(prediction, convert_to_tensor=True)
            similarity = float(util.cos_sim(emb_pred, emb_ref)[0])

            if similarity >= 0.50:
                label = "Grounded"
            elif similarity >= 0.30:
                label = "Partially Hallucinated"
            else:
                label = "Hallucinated"

            # Print results
            print(f"\nQ: {question}")
            print(f"Generated Answer: {prediction}")
            print(f"Reference Answer: {reference}")
            print(f"BLEU-2: {bleu:.4f}, ROUGE-L: {rouge_l:.4f}, F1: {f1:.4f}")
            print(f"Semantic Similarity: {similarity:.4f}")
            print(f"HALLUCINATION LABEL (semantic): {label}")

            print("\nRetrieved Context Chunks (for analysis):")
            for i, chunk in enumerate(retrieved_chunks, start=1):
                print(f"Chunk {i}: {chunk['text'][:300]}")

            # Save results
            hallucination_data.append({
                "question": question,
                "generated_answer": prediction,
                "reference_answer": reference,
                "bleu_2": bleu,
                "rouge_l": rouge_l,
                "f1": f1,
                "semantic_similarity": similarity,
                "retrieved_chunks": retrieved_chunks,
                "hallucination_label": label
            })

            total_bleu += bleu
            total_rougeL += rouge_l
            total_f1 += f1
            count += 1

        except Exception as e:
            print(f"Error processing question: {question}")
            print(f"Exception: {e}")
            continue

    # Print average scores
    print("\n===== AVERAGE METRICS =====")
    print(f"Average BLEU-2: {total_bleu / count:.4f}")
    print(f"Average ROUGE-L: {total_rougeL / count:.4f}")
    print(f"Average F1 Score: {total_f1 / count:.4f}")

    # Save results
    with open("rag_results.json", "w", encoding="utf-8") as out_file:
        json.dump(hallucination_data, out_file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
