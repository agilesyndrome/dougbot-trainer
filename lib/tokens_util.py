class TokenizerUtils:
    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset_key_for_tokenization = 'text'

        self.max_token_length = self.tokenizer.model_max_length

        # Overlapping by {overlap} tokens can make it easier for the model to train
        self.overlap = 128

        self.tokenizer_options = {
            # "padding": config.get("padding", "max_length"),
            "truncation": False,
            "max_length": 99999,
            #"max_length": config.get("max_length", None),
            "return_attention_mask": False
        }



    def process(self, dataset):

        # Create the tokens
        dataset.ds = dataset.ds.map(self.tokenize, batched=False)


        # Flatten lists of lists
        dataset.ds = dataset.ds.flatten()
        

    # Can we detokenize text? Yes we can!
    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def tokenize(self, data):

        text = None

        if type(data) == str:
            text = data
        else:
            text = data[self.dataset_key_for_tokenization]

        tokens = self.tokenizer(text, **self.tokenizer_options)

        assert 'input_ids' in tokens, f"Tokenizer did not return input_ids: {tokens}"

        input_ids = tokens['input_ids']

        # We need to resize the input_ids to the model's max_length
        # breaking tokens into chunks when necessary
        chunks = []

        for i in range(0, len(input_ids), self.max_token_length - self.overlap):
            chunk_input_ids = input_ids[i:i + self.max_token_length]

            # Pad the chunk if it's shorter than max_length
            if len(chunk_input_ids) < self.max_token_length:
                chunk_input_ids += [self.tokenizer.pad_token_id] * (self.max_token_length - len(chunk_input_ids))

            # Set up input_ids and labels for this chunk
            chunk = {
                "input_ids": chunk_input_ids,
                "labels": [token if token != self.tokenizer.pad_token_id else -100 for token in chunk_input_ids]
            }

            chunks.append(chunk)

        return_value = {"input_ids": [chunk["input_ids"] for chunk in chunks],
            "labels": [chunk["labels"] for chunk in chunks]}
        # Return chunks as individual rows for the dataset
        return return_value


