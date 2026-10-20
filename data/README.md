# Datasets and Provenance

This directory contains evaluation datasets used to benchmark semantic caching strategies.

## Bitext Customer Support Dataset

- **Dataset Name**: Bitext Customer Support LLM Chatbot Training Dataset
- **Upstream Source**: [Hugging Face (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`)](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- **License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Domain**: Customer service intents, queries, and conversational responses.

### Schema and Columns

| Column | Type | Description |
|---|---|---|
| `flags` | string | Linguistic and stylistic tag markers (e.g., colloquialisms, abbreviations). |
| `instruction` | string | User utterance / customer request text. |
| `category` | string | High-level domain category (e.g., `ACCOUNT`, `ORDER`, `REFUND`, `FEEDBACK`). |
| `intent` | string | Granular task intent label (e.g., `check_refund_policy`, `place_order`, `registration_problems`). |
| `response` | string | Reference customer-support agent response text. |

### Included Files

1. **`bitext_sample_500.csv`** (~348 KB):
   - A representative 500-sample stratified/random subset of the parent corpus.
   - Included in version control under CC BY 4.0 to permit immediate, reproducible local benchmarking and exploratory caching experiments without requiring external data downloads.

2. **`bitext_customer_support.csv`** (~19.2 MB, excluded via `.gitignore`):
   - The full corpus containing thousands of multi-turn customer support dialogues.
   - Excluded from the Git repository to minimize clone weight and avoid storing large binaries in version history.
   - To obtain the full corpus, download it directly from the upstream Hugging Face link above.

## NLTK Movie Reviews Dataset

- **Source**: NLTK corpus (`nltk.corpus.movie_reviews`), Pang & Lee (2004).
- **Format**: 2,000 sentiment-labeled movie reviews (1,000 positive, 1,000 negative).
- **Access**: Downloaded automatically or via `python -c "import nltk; nltk.download('movie_reviews')"` for the offline simulation benchmarks.
