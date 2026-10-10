from transformers import AutoTokenizer

# define the sentence to tokenize
sentence = "Hello world!"

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
token_ids = tokenizer(sentence).input_ids
print(token_ids)

for token_id in token_ids:
    print(tokenizer.decode(token_id))