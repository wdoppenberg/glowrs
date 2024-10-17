import json
from sentence_transformers import SentenceTransformer

SENTENCES = [
	"The cat sits outside",
	"A man is playing guitar",
	"I love pasta",
	"The new movie is awesome",
	"The cat plays in the garden",
	"A woman watches TV",
	"The new movie is so great",
	"Do you like pizza?",
	"The cat sits",
]

MODELS = [
	"jinaai/jina-embeddings-v2-small-en",
	"sentence-transformers/all-MiniLM-L6-v2",
	"sentence-transformers/multi-qa-distilbert-cos-v1",
]


def generate_examples(model: str) -> list:
	model = SentenceTransformer(model, trust_remote_code=True)
	embeddings = model.encode(SENTENCES, normalize_embeddings=False, batch_size=len(SENTENCES))
	return [
		{"sentence": sentence, "embedding": embedding.tolist()} for sentence, embedding in zip(SENTENCES, embeddings)
	]


if __name__ == "__main__":
	out = {
		"fixtures": [
			{
				"core": m,
				"examples": generate_examples(m)

			} for m in MODELS]
	}

	with open("crates/glowrs/tests/fixtures/embeddings/examples.json", "w") as f:
		json.dump(out, f)
