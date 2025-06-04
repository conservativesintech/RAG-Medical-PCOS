class DocumentQA:
    def __init__(self, retriever, generator, max_context_length=2000):
        """
        retriever: an object with a query(query_text, k) method that returns relevant document chunks.
        generator: an object with generate_response(query_text, context) method that returns answer string.
        max_context_length: max characters for combined context to feed the generator.
        """
        self.retriever = retriever
        self.generator = generator
        self.max_context_length = max_context_length

    def get_answer(self, query_text, k=10):
        contexts = self.retriever.query(query_text, k=k)
        if not contexts:
            return "No relevant context found."
        combined_context = "\n\n".join([context["text"] for context in contexts])
        if len(combined_context) > self.max_context_length:
            combined_context = combined_context[:self.max_context_length]
            
        return self.generator.generate_response(query_text, combined_context)

    def hallucination_test(self, query_text, k=10):
        """
        Returns a dict with the question, generated answer, and the top-k retrieved document chunks.
        Useful for manual hallucination analysis.
        """
        contexts = self.retriever.query(query_text, k=k)
        if not contexts:
            return {
                "query": query_text,
                "answer": "No relevant context found.",
                "retrieved_chunks": []
            }

        combined_context = "\n\n".join([context["text"] for context in contexts])
        if len(combined_context) > self.max_context_length:
            combined_context = combined_context[:self.max_context_length]
        
        answer = self.generator.generate_response(query_text, combined_context)

        # Prepare the chunks text for manual tracing
        retrieved_texts = [{
            "text": context["text"],
            "file": context["file"],
            "pages": context["pages"]
        } for context in contexts]

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_chunks": retrieved_texts
        }
    