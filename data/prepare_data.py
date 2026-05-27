# data/prepare_data.py
""" preprocesses or formats data """


def format_chat_example(messages, tokenizer):
    """ turns one chat conversation into a string (just text) """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize = False,           # return the text first and tokenize later
            add_generation_prompt = False,
        )

    parts = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")

    return "\n\n".join(parts)


def maybe_add_eos(text, tokenizer, cfg):
    """ appends EOS between examples before packing """
    if not getattr(cfg, "packing_add_eos_between_examples", True):
        return text
    if tokenizer.eos_token is None:
        return text

    return text + tokenizer.eos_token


def tokenize_dataset(dataset, tokenizer, cfg):
    """ tokenizes all examples for causal language modeling """
                                        # packs samples into fixed-length blocks
    packing_enabled = getattr(cfg, "packing_enabled", True)
                                        # for testing GPU max memory; if False,
                                        # padding is done later dynamically per batch
    optional_padding = "max_length" if getattr(cfg, "pad_to_max_length", False) else False

    def tokenize_fn(batch):
        """ tokenizes one batch of examples; just needed for tokenize_dataset() locally """
        if cfg.input_format == "chat":
            texts = [
                format_chat_example(messages, tokenizer)
                for messages in batch[cfg.messages_column_name]
            ]
        elif cfg.input_format == "text":
            texts = batch[cfg.text_column_name]
        else:
            raise ValueError(f"unknown data format: {cfg.input_format}")

        return tokenizer(
            texts,
            truncation = not packing_enabled,
            max_length = None if packing_enabled else cfg.max_sequence_length,
            padding = False if packing_enabled else optional_padding,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched = True,
        batch_size = getattr(cfg, "preprocessing_batch_size", 1000),
        num_proc = getattr(cfg, "preprocessing_num_proc", 1),
        remove_columns = dataset.column_names,
    )

    if packing_enabled:
        tokenized = pack_tokenized_dataset(tokenized, cfg)
    
    return tokenized


def pack_tokenized_dataset(dataset, cfg):
    """ concatenates tokenized examples into fixed-length token blocks """
    block_size = getattr(cfg, "packing_block_size", cfg.max_sequence_length)
    drop_remainder = getattr(cfg, "packing_drop_remainder", True)

    def pack_fn(batch):
        """ packs tokenized examples into fixed-length blocks """
        concatenated = {}
        for key in batch:
            concatenated[key] = []
            for values in batch[key]:
                concatenated[key].extend(values)

        total_length = len(concatenated["input_ids"])

        if drop_remainder:
            total_length = (total_length // block_size) * block_size

        result = {}
        for key, values in concatenated.items():
            result[key] = [
                values[index : index + block_size]
                for index in range(0, total_length, block_size)
            ]

        return result

    return dataset.map(
        pack_fn,
        batched = True,
        batch_size = getattr(cfg, "preprocessing_batch_size", 1000),
        num_proc = getattr(cfg, "preprocessing_num_proc", 1),
        remove_columns = dataset.column_names,
    )