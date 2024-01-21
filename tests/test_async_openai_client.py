from openai import AsyncOpenAI
from time import time
import asyncio

client = AsyncOpenAI(
	api_key="sk-something",
	base_url="http://127.0.0.1:3000/v1"
)


async def create_embeddings():
	start = time()
	embeddings = await client.embeddings.create(
		input=["This is a sentence that requires an embedding"] * 10,
		model="jina-embeddings-v2-base-en"
	)
	print(len(embeddings.data[0].embedding))
	print(f"Done in {time() - start}")


async def main():
	await asyncio.gather(*(create_embeddings() for _ in range(15)))


asyncio.run(main())
