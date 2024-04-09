import openai
from openai import AsyncOpenAI
from httpx import AsyncClient
from time import time
import asyncio

from openai.types import CreateEmbeddingResponse

openai_client = AsyncOpenAI(
	api_key="sk-something",
	base_url="http://127.0.0.1:3000/v1"
)


async def create_embeddings() -> tuple[CreateEmbeddingResponse, ...]:
	embeddings = await openai_client.embeddings.create(
		input=["This is a sentence that requires a model and is quite long for a normal sentence"] * 5,
		model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
	)

	embeddings_jina = await openai_client.embeddings.create(
		input=["This is a sentence that requires a model and is quite long for a normal sentence"] * 5,
		model="jinaai/jina-embeddings-v2-base-en",
	)

	return embeddings, embeddings_jina


async def call_health() -> None:
	# Call the /health endpoint 100 times
	async with AsyncClient() as client:
		for _ in range(100):
			response = await client.get("http://127.0.0.1:3000/health")

			response.raise_for_status()


async def call_model_list() -> None:
	model_list = await openai_client.models.list()
	print(model_list)


async def main():
	start = time()

	await asyncio.gather(call_health(), *(create_embeddings() for _ in range(10)))

	try:
		await openai_client.embeddings.create(
			input=["This is a sentence that requires a model and is quite long for a normal sentence"] * 5,
			model="does-not-exist",
		)
	except openai.NotFoundError as e:
		print("Correctly raised NotFoundError")

	await call_model_list()

	print(f"Done in {time() - start}")


asyncio.run(main())
