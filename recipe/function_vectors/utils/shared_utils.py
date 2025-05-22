import typing
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import wandb


@dataclass
class EvalDataResults:
    sentences: typing.List[str] = field(default_factory=list)
    targets: typing.List[str] = field(default_factory=list)
    skipped_indices: typing.Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    logits: torch.Tensor | None = None
    nlls: typing.List[float] | None = None
    scores: typing.List[float] | None = None
    strings: typing.List[str] | None = None

    def __len__(self):
        return len(self.sentences)

    def append(self, sentence, target):
        self.sentences.append(sentence)
        self.targets.append(target)

    def clone(self):
        return EvalDataResults(sentences=self.sentences.copy(), targets=self.targets.copy())


# def tokenizer_padding_side_token(padding_side=None):
def tokenizer_padding_side_token(func):
    def wrapper(*args, **kwargs):
        if "tokenizer" not in kwargs:
            raise ValueError("tokenizer must be provided as a keyword argument.")

        tokenizer = kwargs["tokenizer"]
        generate_str = kwargs.get("generate_str", False)

        initial_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left" if generate_str else "right"
        # tokenizer.padding_side = padding_side if padding_side is not None else ("left" if generate_str else "right")

        pad_token_set = False
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # tokenizer sets the pad token id automatically when the pad token is set
            # tokenizer.pad_token_id = tokenizer.eos_token_id
            pad_token_set = True

        try:
            return func(*args, **kwargs)

        finally:
            if pad_token_set:
                tokenizer.pad_token = None
                # tokenizer.pad_token_id = None

            tokenizer.padding_side = initial_padding_side

    return wrapper

    # return inner_padding_side_token


def wandb_log(phase: str, **kwargs):
    if wandb.run is not None:
        wandb.log({phase: phase, **kwargs})
