# !pip install openai

from openai import OpenAI
from time import time

client = OpenAI(
	api_key="sk-something",
	base_url="http://127.0.0.1:3000/v1"
)

start = time()
print(client.embeddings.create(
	input=["This is a sentence that requires an core"] * 50,
	model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
))

print(f"Done in {time() - start}")
