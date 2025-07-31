from document_processing import load_document
from embeddings import EmbeddingEngine, FAISSVectorStore, ClauseQueryEngine
import os
from huggingface_hub import InferenceClient
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import requests
import hashlib

os.environ['HF_TOKEN'] = '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'

embedding_model = 'BAAI/bge-base-en-v1.5'
# embedding_model = 'mixedbread-ai/mxbai-embed-large-v1'
llm_model="meta-llama/Llama-3.3-70B-Instruct"
hf_provider = "nebius"
API_KEY = "5d707fdbaa86dd536a7dda05d85613028623a6fadbc2f91c6d2d05f2de5eed41"
# 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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

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

@app.post("/hackrx/run", response_model=QueryResponse)
# async def answer_batch_questions(document_url, questions_list, top_k=5):
async def answer_batch_questions(request: Request, body: QueryRequest):
    
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer ") or auth.split(" ")[1] != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    # print(vectors.shape)
    try:
        _ = ingest_document(body.documents, overwrite=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

    retriever = ClauseQueryEngine(model_name=embedding_model)
    client = InferenceClient(
        provider=hf_provider,
        api_key=os.environ["HF_TOKEN"],
    )

    # system="You are a helpful assistant answering questions from contracts and policies."
    answers = []
    for question in body.questions:
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

        response = completion.choices[0].message.content
        # answer = response.strip().replace('\n', ' ')
        answers.append(response)
        # print(completion.choices[0].message.content)

    return JSONResponse(content={"answers": answers})

