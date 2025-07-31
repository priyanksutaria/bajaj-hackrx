from document_processing import load_document
from embeddings import EmbeddingEngine, FAISSVectorStore, ClauseQueryEngine
import os
from huggingface_hub import InferenceClient

os.environ['HF_TOKEN'] = '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'

embedding_model = 'BAAI/bge-base-en-v1.5'
# embedding_model = 'mixedbread-ai/mxbai-embed-large-v1'
llm_model="meta-llama/Llama-3.3-70B-Instruct"
hf_provider = "nebius"
# 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

def ingest_document(source_path_or_url, overwrite):
    data = load_document(source_path_or_url)
    clauses = data["clauses"]

    texts = [f"passage: {clause['text']}" for clause in clauses] # passage is important for the model of embeddings to work
    metadata_list = clauses  # Each clause has text, page, position

    embedder = EmbeddingEngine(model_name=embedding_model)
    vectors = embedder.encode(texts)

    store = FAISSVectorStore(dim=vectors.shape[1], overwrite=overwrite)
    store.add(vectors, metadata_list)
    store.save()

    print(f"Ingested {len(texts)} clauses into FAISS index.")
    
    return vectors

def build_prompt(question, clauses):
    clause_texts = "\n\n".join([f"Clause {i+1}: {c['text']}" for i, c in enumerate(clauses)])

    full_prompt = f"""
        You are a helpful assistant answering questions from contracts and policies.

        A user has asked the following question:
        "{question}"

        Use only the clauses below to answer it:

        {clause_texts}

        Give a clear, concise answer based only on the provided clauses.
        Return ONLY the answer as a plain sentence. Do NOT include any explanations or rationale. No JSON, just a direct answer.

        """

    return {
        "type": "text",
        "text": full_prompt
    }

def answer_batch_questions(document_url, questions_list, top_k=5):
    
    vectors = ingest_document(document_url, overwrite=True)
    print(vectors.shape)

    retriever = ClauseQueryEngine(model_name=embedding_model)
    client = InferenceClient(
        provider=hf_provider,
        api_key=os.environ["HF_TOKEN"],
    )

    # system="You are a helpful assistant answering questions from contracts and policies."
    answers = []
    for question in questions_list:
        matched_clauses = retriever.search(question, top_k=5)
        
        prompt = build_prompt(question, matched_clauses)

        completion = client.chat.completions.create(
            model=llm_model,
            # messages=[
            #     {"role": "system", "content": system},
            #     {"role": "user", "content": prompt}
            # ],

            messages=[
                {
                    "role": "user",
                    "content": prompt['text']
                }
            ],
            temperature=0.3
        )

        # # Clean and add answer
        response = completion.choices[0].message.content
        # answer = response.strip().replace('\n', ' ')
        answers.append(response)
        # print(completion.choices[0].message.content)

    return {"answers": answers}


def clauses_check(document_url, questions_list):
    
    vectors = ingest_document(document_url, overwrite=True)
    print(vectors.shape)

    retriever = ClauseQueryEngine(model_name=embedding_model)

    for question in questions_list:
        matched_clauses = retriever.search(question, top_k=5)
        print(f"\n {question}: \n")

        for i, clause in enumerate(matched_clauses, 1):
            print(f"\nMatch #{i}")
            print(f"Page: {clause.get('page', 'N/A')} | Position: {clause.get('position', 'N/A')}")
            print(f"Text: {clause['text']}")
        print("\n")


if __name__ == "__main__":

    # engine = ClauseQueryEngine(model_name=embedding_model)

    # query = "How does the policy define a 'Hospital'?"
    # results = engine.search(query, top_k=5)

    # for i, clause in enumerate(results, 1):
    #     print(f"\n Match #{i}")
    #     print(f"Page: {clause.get('page', 'N/A')} | Position: {clause.get('position', 'N/A')}")
    #     print(f"Text: {clause['text']}")

    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]

    # questions = [
    #     "What is the No Claim Discount (NCD) offered in this policy?"
    # ]

    # clauses_check('policy.pdf', questions)
    answer = answer_batch_questions("https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D", questions)




