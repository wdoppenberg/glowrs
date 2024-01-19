# !pip install openai

from openai import OpenAI
from time import time

client = OpenAI(
	api_key="something",
	base_url="http://127.0.0.1:3000/v1"
)

start = time()
print(client.embeddings.create(
	input=["This is a sentence that is quite a bit longer"] * 40,
	model="something"
))

print(f"Done in {time() - start}")