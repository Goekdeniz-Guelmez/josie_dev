import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    Any,
    Tuple,
)

import tiktoken
import torch
from tiktoken.load import load_tiktoken_bpe

logger = getLogger(__name__)

TIKTOKEN_MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACES_CHARS = 25_000


class Tokenizer:
    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 256
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str ='tokenizer.model'):
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",  # end of message
            "<|eot_id|>",  # end of turn
            "<|python_tag|>",
            "<|im_audio_start|>",
            "<|audio|>",
            "<|im_audio_end|>",
            "<|im_vision_start|>",
            "<|vision|>",
            "<|im_vision_end|>",
            "<|audio_conversation|>"
        ]
        reserved_tokens = [
            f"<|reserved_special_token_{2 + i}|>"
            for i in range(self.num_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.vocab_size: int = num_base_tokens + len(special_tokens)
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.eot_id: int = self.special_tokens["<|eot_id|>"]
        self.eom_id: int = self.special_tokens["<|eom_id|>"]
        self.python_tag_id = self.special_tokens["<|python_tag|>"]
        self.pad_id: int = self.special_tokens["<|finetune_right_pad_id|>"]
        self.audio_conversation: int = self.special_tokens["<|audio_conversation|>"]
        self.stop_tokens = [
            self.special_tokens["<|eom_id|>"],
            self.special_tokens["<|eot_id|>"],
        ]

    def _pad_sequence(
        self, 
        sequences: List[List[int]], 
        padding_value: int = None
    ) -> Tuple[List[List[int]], List[int]]:
        if padding_value is None:
            padding_value = self.pad_id

        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        attention_mask = []

        for seq in sequences:
            padding_length = max_len - len(seq)
            padded_seq = seq + [padding_value] * padding_length
            mask = [1] * len(seq) + [0] * padding_length
            padded_sequences.append(padded_seq)
            attention_mask.append(mask)

        return padded_sequences, attention_mask

    def encode(
        self,
        texts: Union[str, List[str]],
        *,
        bos: bool = True,
        eos: bool = False,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
        return_tensors: Optional[Literal["pt", "np"]] = None,
        padding: bool = True,
        return_attention_mask: bool = True,
    ) -> Union[
        List[int],
        List[List[int]],
        Dict[str, Union[torch.Tensor, Any]],
    ]:
        if allowed_special is None:
            allowed_special = set()

        if isinstance(texts, str):
            tokens = self._encode_single(
                texts, bos, eos, allowed_special, disallowed_special
            )
            if return_tensors:
                tokens = [tokens]  # Make it a batch of 1
            else:
                return tokens  # Return as single sequence

        else:
            tokens = [
                self._encode_single(text, bos, eos, allowed_special, disallowed_special)
                for text in texts
            ]

        attention_mask = None
        if len(tokens) > 1 and padding:
            tokens, attention_mask = self._pad_sequence(tokens)

        if return_tensors:
            if return_tensors == "pt":
                output_tokens = torch.tensor(tokens, dtype=torch.long)
                if attention_mask is not None and return_attention_mask:
                    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            elif return_tensors == "np":
                import numpy as np
                output_tokens = np.array(tokens, dtype=np.int64)
                if attention_mask is not None and return_attention_mask:
                    attention_mask = np.array(attention_mask, dtype=np.int64)

            output = {"input_ids": output_tokens}
            if attention_mask is not None and return_attention_mask:
                output["attention_mask"] = attention_mask
            return output

        return tokens

    def _encode_single(
        self,
        s: str,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]],
        disallowed_special: Union[Literal["all"], Collection[str]],
    ) -> List[int]:
        assert isinstance(s, str)
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(
        self, 
        t: Union[
            Sequence[int],
            List[List[int]],
            torch.Tensor,
            Any
        ],
        skip_special_tokens: bool = False,
    ) -> Union[str, List[str]]:
        if isinstance(t, torch.Tensor):
            if t.dim() == 1:
                t = [t.tolist()]
            else:
                t = t.tolist()
        elif hasattr(t, '__array__'):  # NumPy array
            if t.ndim == 1:
                t = [t.tolist()]
            else:
                t = t.tolist()
        elif isinstance(t, list) and not isinstance(t[0], list):
            t = [t]

        decoded = []
        for seq in t:
            if skip_special_tokens:
                seq = [token for token in seq if token not in self.special_tokens.values()]
            decoded.append(self.model.decode(cast(List[int], seq)))

        if len(decoded) == 1:
            return decoded[0]
        return decoded

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]