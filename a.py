# Cell 1: Imports & Environment Setup
import os
import time
import json
import boto3
from dotenv import load_dotenv

# LangChain & AI Libraries
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

# Pinecone & Evaluation
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder


# Load Environment Variables
load_dotenv(override=True)

print("✅ Libraries loaded. Environment verified.")



# Cell 2: Smart Initialization & Duplicate Check
# Configuration
file_path = "SBIhomeinsurance_home.pdf" # Make sure this matches your file name
index_name = "sbi-home-insurance-rag-hybrid" # Using your existing hybrid index name

# 1. Connect to Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# 2. Check if Index Exists
existing_indexes = [index.name for index in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"⚠️ Index '{index_name}' not found. Creating it...")
    pc.create_index(
        name=index_name,
        dimension=1024, # Titan v2
        metric="dotproduct", # Required for Hybrid
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(20) # Wait for init
    print("✅ Index created successfully.")
else:
    print(f"✅ Index '{index_name}' already exists.")

# 3. Connect to the Index
index = pc.Index(index_name)

# 4. Check if File is Already Ingested (The "Smart" Check)
# We perform a dummy query filtering by this specific source file
print(f"🔍 Checking if '{file_path}' is already in the database...")

# We use a dummy vector just to trigger the metadata filter
dummy_vector = [0.0] * 1024 
check_response = index.query(
    vector=dummy_vector,
    top_k=1,
    filter={"source": file_path},
    include_metadata=False
)

if len(check_response['matches']) > 0:
    print(f"✅ File '{file_path}' detected in Pinecone.")
    print("🚀 SKIPPING Docling & Embeddings to save cost.")
    should_ingest = False
else:
    print(f"⚠️ File '{file_path}' NOT found in Pinecone.")
    print("⚙️ Proceeding with Ingestion...")
    should_ingest = True






# Cell 3: Load & Chunk (Conditional)
final_chunks = []

if should_ingest:
    print(f"📄 Starting Docling processing for {file_path}...")
    
    # A. Load with Docling (Export to Markdown)
    loader = DoclingLoader(
        file_path=file_path,
        export_type=ExportType.MARKDOWN
    )
    docs = loader.load()
    print("✅ PDF Loaded via Docling.")

    # B. Split by Headers (Level 1)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(docs[0].page_content)
    
    # C. Split by Size (Level 2)
    chunk_size = 1000
    chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    final_chunks = text_splitter.split_documents(md_header_splits)

    # D. Add Metadata Tags (Crucial for Smart Indexing)
    for chunk in final_chunks:
        chunk.metadata["source"] = file_path # Used for filtering later
        # We also keep the 'text' in metadata for Hybrid retrieval
        chunk.metadata["text"] = chunk.page_content 
    
    print(f"✅ Chunking Complete. Created {len(final_chunks)} chunks.")
    print("Sample Metadata:", final_chunks[0].metadata)

else:
    print("⏭️ Skipping Loading & Chunking (Data already exists).")





# Cell 4: Hybrid Embedding & Upsert (Conditional)
import boto3
from langchain_aws import BedrockEmbeddings
from pinecone_text.sparse import BM25Encoder

# 1. Initialize AWS Bedrock Embeddings (Need this for both Ingestion AND Querying)
boto3_session = boto3.Session()
bedrock_client = boto3_session.client("bedrock-runtime", region_name="us-east-1")

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock_client
)

# 2. Initialize BM25 Encoder
bm25 = BM25Encoder()
bm25_filename = "bm25_values.json"

if should_ingest:
    print("⚙️ Generatings Embeddings & Upserting...")
    
    # A. Fit BM25 on the new text
    chunk_texts = [chunk.page_content for chunk in final_chunks]
    bm25.fit(chunk_texts)
    bm25.dump(bm25_filename) # Save for future use
    print("✅ BM25 Encoder fitted and saved.")
    
    # B. Generate Vectors & Upsert
    vectors_to_upsert = []
    
    print(f"Generating vectors for {len(final_chunks)} chunks...")
    for i, chunk in enumerate(final_chunks):
        # 1. Dense Vector (Titan)
        dense_vec = embeddings.embed_query(chunk.page_content)
        
        # 2. Sparse Vector (BM25)
        sparse_vec = bm25.encode_documents(chunk.page_content)
        
        # 3. Create ID (Unique based on source + index)
        # We use a simple hash or index. Here index 'i' is fine for this run.
        # Ideally, hash the text to avoid dupes, but for now:
        vector_id = f"{file_path}_{i}"
        
        vectors_to_upsert.append({
            "id": vector_id,
            "values": dense_vec,
            "sparse_values": sparse_vec,
            "metadata": chunk.metadata # Includes 'source' and 'text'
        })
        
    # C. Batch Upsert to Pinecone
    batch_size = 50
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"   Uploaded batch {i} to {i+batch_size}")
        
    print("✅ Ingestion Complete.")

else:
    # If we skipped ingestion, we MUST load the BM25 model from disk
    # so we can still run queries.
    if os.path.exists(bm25_filename):
        bm25.load(bm25_filename)
        print("✅ Skipped Ingestion. Loaded existing BM25 params from file.")
    else:
        print("⚠️ Warning: BM25 file not found. You might need to re-ingest if retrieval fails.")




# Cell 5: Setup Retrieval & Re-ranking Engines
from typing import List

# 1. Define the Bedrock Cohere Re-ranker Class
class BedrockCohereReranker:
    def __init__(self, region_name="us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.model_id = "cohere.rerank-v3-5:0"

    def rerank(self, query: str, docs: List[str], top_n: int = 5):
        # Docs must be a list of strings for the API
        if not docs: return []
        
        request_body = {
            "query": query, 
            "documents": docs, 
            "top_n": top_n, 
            "api_version": 2
        }
        
        try:
            response = self.client.invoke_model(modelId=self.model_id, body=json.dumps(request_body))
            response_body = json.loads(response['body'].read())
            results = response_body.get("results", [])
            return results # Returns list of {'index': int, 'relevance_score': float}
        except Exception as e:
            print(f"⚠️ Rerank Error: {e}")
            # Fallback: return indices 0..top_n
            return [{"index": i, "relevance_score": 0.0} for i in range(min(len(docs), top_n))]

# Initialize the Reranker
reranker = BedrockCohereReranker()
print("✅ Cohere Re-ranker Initialized.")

# 2. Define the "Intelligent Retrieval" Function
# This combines Hybrid Search (Pinecone) + Re-ranking (Cohere)
def intelligent_retrieval(query: str) -> str:
    print(f"🔎 Searching for: '{query}'")
    
    # A. Hybrid Search in Pinecone (Top 25)
    dense_vec = embeddings.embed_query(query)
    # Note: If you want strict keyword matching, enable the line below:
    # sparse_vec = bm25.encode_queries(query) 
    
    results = index.query(
        vector=dense_vec,
        # sparse_vector=sparse_vec, # Uncomment if passing sparse values
        top_k=25,
        include_metadata=True
    )
    
    # Extract just the text from the matches
    raw_docs = [match['metadata']['text'] for match in results['matches']]
    
    if not raw_docs:
        return ""

    # B. Re-ranking (Filter 25 -> Top 5)
    rerank_results = reranker.rerank(query, raw_docs, top_n=5)
    
    # C. Format the Top 5 for the LLM
    top_docs_text = []
    for res in rerank_results:
        idx = res['index']
        top_docs_text.append(raw_docs[idx])
        
    return "\n\n".join(top_docs_text)

print("✅ Retrieval Logic Defined.")





# Cell 5.5: Initialize anthropic.claude-3-5-haiku Model

from langchain_aws import ChatBedrock

# We use the US Cross-Region Inference Profile for Llama 3.1
llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",  ## us.meta.llama3-1-70b-instruct-v1:0
    client=bedrock_client, # We defined this client in Cell 4
    model_kwargs={"temperature": 0.1, "max_tokens": 512} # max_tokens": 2048
)

print("✅ anthropic.claude-3-5-haiku Model Initialized.")




# Cell 6: LLM Chain Setup
# 1. Define the Prompt
# We strictly tell the LLM to use ONLY the provided context.
prompt_template = """
You are an expert Insurance Assistant. Use the following pieces of retrieved context to answer the question.
If the answer is not in the context, just say that you don't know. Do not try to make up an answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

prompt = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

# 2. Define the Chain
# This pipeline does: Take Query -> Get Smart Context -> Format Prompt -> Run Llama 3 -> Parse String
rag_chain_final = (
    {
        "context": RunnableLambda(intelligent_retrieval), # Uses our Hybrid + Rerank function
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG Chain (Production Ready) Created.")

# 3. Quick Sanity Check
# Let's run a simple test to make sure the chain flows correctly

test_q = "What specific exclusions apply to loss caused by Subsidence?"



print(f"\n🧪 Sanity Check Query: '{test_q}'")
print("-" * 40)
print(rag_chain_final.invoke(test_q))