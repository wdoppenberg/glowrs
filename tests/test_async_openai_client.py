from openai import AsyncOpenAI
from httpx import AsyncClient
from time import time
import asyncio

from openai.types import CreateEmbeddingResponse

openai_client = AsyncOpenAI(
	api_key="sk-something",
	base_url="http://127.0.0.1:3000/v1"
)


async def create_embeddings() -> CreateEmbeddingResponse:
	embeddings = await openai_client.embeddings.create(
		input=["This is a sentence that requires a model and is quite long for a normal sentence"] * 10,
		model="jina-embeddings-v2-base-en"
	)

	return embeddings


async def call_health() -> None:
	# Call the /health endpoint 100 times
	async with AsyncClient() as client:
		for _ in range(100):
			response = await client.get("http://127.0.0.1:3000/health")

			response.raise_for_status()


async def main():
	start = time()

	await asyncio.gather(call_health(), *(create_embeddings() for _ in range(100)))

	print(f"Done in {time() - start}")


asyncio.run(main())
