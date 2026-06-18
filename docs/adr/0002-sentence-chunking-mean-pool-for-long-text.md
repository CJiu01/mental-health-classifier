# Sentence chunking with mean pooling for long-text input in Ch.11

Ch.11 splits multi-sentence input into individual sentences (nltk sent_tokenize), encodes each with the fine-tuned MentalBERT, then mean-pools the resulting vectors before classification. This keeps each sentence within the short-text distribution that matches the training data (dair-ai/emotion is single-sentence Twitter posts).

## Considered Options

- **Truncate at 512 tokens**: simple but loses information beyond the first ~20 sentences and mismatches the per-sentence training distribution.
- **Max pooling (pick highest-risk sentence)**: intuitive for crisis detection but requires running the full classifier per sentence, not just the encoder.
- **Mean pooling of per-sentence encodings** *(chosen)*: matches training distribution, handles arbitrary length, clean implementation aligned with Ch.10 bi-encoder concept.
