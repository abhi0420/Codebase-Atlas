import chromadb
from chromadb.utils import embedding_functions
import os

from dotenv import load_dotenv

load_dotenv()

sample_docs = [
    "The insurance policy is valid for one year from the date of issue.",
    "The insured must notify the insurer of any changes in risk factors.",
    "Premiums must be paid in full to maintain coverage under the policy.",
    "In the event of a claim, the insured must provide all necessary documentation.",
    "In case all documents are correct, the claim will be processed within 30 days."
]

chroma_client = chromadb.Client()
embedding = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-ada-002")

collection  = chroma_client.create_collection(name="test_collection", embedding_function=embedding)

collection.add(documents=sample_docs, ids=[str(i) for i in range(len(sample_docs))])

query = "What is the duration of the insurance policy?"

results = collection.query(query_texts=[query], n_results=3)

print("Query Results:")
print(results["documents"][0][0])


