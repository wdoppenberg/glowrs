from openai import AsyncOpenAI
from time import time
import asyncio

from openai.types import CreateEmbeddingResponse

client = AsyncOpenAI(
	api_key="sk-something",
	base_url="http://127.0.0.1:3000/v1"
)


async def create_embeddings() -> CreateEmbeddingResponse:
	start = time()
	embeddings = await client.embeddings.create(
		input=["This is a sentence that requires an embedding and is quite long for a normal sentence"] * 10,
		model="jina-embeddings-v2-base-en"
	)
	print(len(embeddings.data[0].embedding), embeddings.usage)
	print(f"Done in {time() - start}")

	return embeddings


async def main():
	e = await asyncio.gather(*(create_embeddings() for _ in range(10000)))

	print(len(e))

asyncio.run(main())
