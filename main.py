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
from bert_score import score as bert_score
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--retrieval_k", type=int, default=5, help="Top K documents to retrieve")
args = parser.parse_args()

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

    with open("pcos_qna_evidence_based.json", "r", encoding="utf-8") as file:
        pcos_qna = json.load(file)

    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    total_bleu, total_rougeL, total_f1, total_bert_score = 0, 0, 0, 0
    count = 0
    result_data = []

    for question, reference in pcos_qna.items():
        try:
            result = qa_system.get_answer(question, k=args.retrieval_k)
            prediction = result["answer"]
            retrieved_chunks = result["retrieved_chunks"]  # Optional: use for debugging/hallucination tracing

            if not prediction or prediction.lower().strip() in ["", "i don't know", "not found in the passage."]:
                prediction = "Not found in the passage."

            # BLEU-2 Score
            ref_tokens = word_tokenize(reference.lower())
            pred_tokens = word_tokenize(prediction.lower())
            bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5), smoothing_function=smooth)

            # ROUGE-L Score
            rouge_scores = rouge.score(reference, prediction)
            rouge_l = rouge_scores['rougeL'].fmeasure

            # F1 Score
            f1 = compute_f1(reference, prediction)

            # Cosine Similarity
            with torch.no_grad():
                emb_ref = embed_model.encode(reference, convert_to_tensor=True)
                emb_pred = embed_model.encode(prediction, convert_to_tensor=True)
            cosine_similarity = float(util.cos_sim(emb_pred, emb_ref)[0])

            # BERTScore
            P, R, bert_f1 = bert_score([prediction], [reference], lang="en", rescale_with_baseline=True, model_type='bert-base-uncased')
            bertscore = bert_f1[0].item()

            # Label based on semantic cosine_similarity
            if cosine_similarity >= 0.75:
                label = "Grounded"
            elif cosine_similarity >= 0.50:
                label = "Partially Grounded"
            else:
                label = "Hallucinated"

            # Print results
            print(f"\nQ: {question}")
            print(f"Generated Answer: {prediction}")
            print(f"Reference Answer: {reference}")
            print(f"BLEU-2: {bleu:.4f}, ROUGE-L: {rouge_l:.4f}, F1: {f1:.4f}, BERTScore: {bertscore:.4f}")
            print(f"Cosine Similarity: {cosine_similarity:.4f}")
            print(f"HALLUCINATION LABEL (semantic): {label}")

            result_data.append({
                "question": question,
                "generated_answer": prediction,
                "reference_answer": reference,
                "bleu_2": bleu,
                "rouge_l": rouge_l,
                "f1": f1,
                "bert_score": bertscore,
                "cosine_similarity": cosine_similarity,
                "hallucination_label": label,
                "retrieved_chunks": retrieved_chunks  # Optional if you want to save them
            })

            total_bleu += bleu
            total_rougeL += rouge_l
            total_f1 += f1
            total_bert_score += bertscore
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
    print(f"Average BERTScore: {total_bert_score / count:.4f}")
    
    # Save all results
    with open("rag_results.json", "w", encoding="utf-8") as out_file:
        json.dump(result_data, out_file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
