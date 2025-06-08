from sentence_transformers import SentenceTransformer, util

class DocumentQA:
    def __init__(self, retriever, generator, max_context_tokens=None):
        """
        retriever: an object with a query(query_text, k) method.
        generator: an object with .generate_response(query, context) and .tokenizer, .model
        max_context_tokens: (optional) hard cap. If None, will use model.config.max_position_embeddings.
        """
        self.retriever = retriever
        self.generator = generator
        if max_context_tokens is None:
            self.max_context_tokens = generator.model.config.max_position_embeddings
        else:
            self.max_context_tokens = max_context_tokens
        
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _extractive_fallback(self, query, context):
        """Find the sentence in the context most similar to the query."""
        sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 10]
        if not sentences:
            return "No answer found in the passage."
        best_sentence = max(
            sentences,
            key=lambda s: util.cos_sim(
                self.embed_model.encode(query, convert_to_tensor=True),
                self.embed_model.encode(s, convert_to_tensor=True)
            ).item()
        )
        return best_sentence.strip() + "."

    def get_answer(self, query_text, k=5):

        while k >= 1:
            contexts = self.retriever.query(query_text, k=k)
            if not contexts:
                return {
                    "query": query_text,
                    "answer": "No relevant context found.",
                    "retrieved_chunks": [],
                    "k_used": 0
                }

            # Rerank with semantic similarity
            # contexts = sorted(
            #     contexts,
            #     key=lambda c: util.cos_sim(
            #         self.embed_model.encode(query_text, convert_to_tensor=True),
            #         self.embed_model.encode(c['text'], convert_to_tensor=True)
            #     ).item(),
            #     reverse=True
            # )

            query_embedding = self.embed_model.encode(query_text, convert_to_tensor=True)
            chunk_texts = [c['text'] for c in contexts]
            chunk_embeddings = self.embed_model.encode(chunk_texts, convert_to_tensor=True)
            similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]

            for i, sim in enumerate(similarities):
                contexts[i]["similarity"] = sim.item()

            contexts = sorted(contexts, key=lambda x: x["similarity"], reverse=True)
            
            # Add chunks until token budget is exceeded
            tokenizer = self.generator.tokenizer
            total_tokens = 0
            context_tokens = []
            selected_chunks = []
            
            for chunk in contexts:
                chunk_tokens = tokenizer.encode(chunk["text"], add_special_tokens=False)
                if total_tokens + len(chunk_tokens) > self.max_context_tokens - 128:
                    break
                context_tokens.extend(chunk_tokens)
                total_tokens += len(chunk_tokens)
                selected_chunks.append(chunk)

            if selected_chunks:
                break  # Success
            else:
                k -= 1  # Try with fewer chunks
                print(f"Trying with k={k}")

        context = tokenizer.decode(context_tokens, skip_special_tokens=True)
        answer = self.generator.generate_response(query_text, context)

        if not answer or answer.lower() in ["not found in the passage.", "sorry, something went wrong."]:
            answer = self._extractive_fallback(query_text, context)

        retrieved_chunks = [ {
            "text": c["text"],
            "file": c["file"],
            "pages": c["pages"]
        } for c in selected_chunks ]

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "k_used": k  # <-- Added this to track final k
        }