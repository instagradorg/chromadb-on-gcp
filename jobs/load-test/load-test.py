import os
import time
from uuid import uuid4

import chromadb
import matplotlib.pyplot as plt
import pandas as pd
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from dotenv import load_dotenv
from faker import Faker
from langchain_google_vertexai import VertexAIEmbeddings
from tqdm import tqdm

fake = Faker()

load_dotenv()

HOST = os.getenv("CHROMA_DB_HOST")
PORT = os.getenv("CHROMA_DB_PORT")
CHROMA_DB_AUTH_CREDENTIALS = os.getenv("CHROMA_DB_AUTH_CREDENTIALS")
NB_CHUNKS_TO_ADD = 10000
NB_CHUNKS_PER_BATCH = 10
    
embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-004"
)

client = chromadb.HttpClient(
   host=HOST,
   port=PORT,
   ssl=True,
   settings=Settings(
      chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider", 
      chroma_client_auth_credentials=CHROMA_DB_AUTH_CREDENTIALS,
      anonymized_telemetry=False
   )
)

try:
    client.delete_collection(
        name="db-test-collection"
    )
except NotFoundError:
    pass

collection = client.get_or_create_collection(
    name="db-test-collection"
)

 # dimension 712
embedding_to_query = embedding_model.embed_query(text="Hello world")

add_durations = []
query_durations = []

try:

    nb_iteration = int(NB_CHUNKS_TO_ADD / NB_CHUNKS_PER_BATCH)

    for _ in tqdm(range(nb_iteration)):

        chunks = fake.paragraphs(nb=NB_CHUNKS_PER_BATCH)

        embeddings_to_add = embedding_model.embed_documents(chunks)

        start_add = time.time()
        collection.add(
            embeddings=embeddings_to_add,
            documents=chunks,
            ids=[str(uuid4()) for _ in range(NB_CHUNKS_PER_BATCH)],
        )
        end_add = time.time()
        add_durations.append(end_add - start_add)

        start_query = time.time()
        response = collection.query(
            query_embeddings=[embedding_to_query]
        )
        end_query = time.time()
        query_durations.append(end_query - start_query)

finally:

    cumulative_paragraphs = [i * NB_CHUNKS_PER_BATCH for i in range(1, len(add_durations) + 1)]

    add_series = pd.Series(add_durations, index=cumulative_paragraphs)
    query_series = pd.Series(query_durations, index=cumulative_paragraphs)

    add_rolling = add_series.rolling(window=20).mean()
    query_rolling = query_series.rolling(window=20).mean()

    # Plot rolling averages with X-axis as number of paragraphs
    plt.figure()
    plt.plot(add_rolling, label="Vectors uploading duration (Rolling Avg, 20)")
    plt.plot(query_rolling, label="Vectors retrieval duration (Rolling Avg, 20)")
    plt.xlabel("Cumulative Number of Chunks Uploaded")
    plt.ylabel("Duration (seconds)")
    plt.title("Rolling Avg Duration vs. Number of Chunks")
    plt.legend()

    # Save plot
    plot_path = os.path.join(os.path.dirname(__file__), "multi-embedding-duration.png")
    plt.savefig(plot_path)