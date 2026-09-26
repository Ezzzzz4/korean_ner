"""Character-level KLUE NER tokenization and label alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


IGNORE_INDEX = -100
DEFAULT_MAX_LENGTH = 192
SPACE_TOKEN = "[unused0]"


@dataclass(frozen=True)
class AlignedEncoding:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    labels: list[int] | None
    char_indices: list[int | None]
    tokens: list[str]


def _special_id(tokenizer: object, token_name: str, fallback_token: str) -> int:
    value = getattr(tokenizer, token_name, None)
    if value is not None:
        return int(value)
    token_id = tokenizer.convert_tokens_to_ids(fallback_token)
    if token_id is None or token_id < 0:
        raise ValueError(f"tokenizer has no id for {fallback_token}")
    return int(token_id)


def _as_id_list(token_ids: object) -> list[int]:
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def _batch_tokenize_chars(tokenizer: object, chars: Sequence[str]) -> list[tuple[list[int], list[str]]]:
    if not chars:
        return []

    encoded = tokenizer(list(chars), add_special_tokens=False)
    batch_ids = encoded["input_ids"]
    if hasattr(batch_ids, "tolist"):
        batch_ids = batch_ids.tolist()
    if batch_ids and not isinstance(batch_ids[0], (list, tuple)):
        batch_ids = [batch_ids]

    unk_id = _special_id(tokenizer, "unk_token_id", "[UNK]")
    unk_token = getattr(tokenizer, "unk_token", "[UNK]")
    pieces: list[tuple[list[int], list[str]]] = []
    for token_ids in batch_ids:
        ids = _as_id_list(token_ids)
        if not ids:
            pieces.append(([unk_id], [unk_token]))
            continue
        tokens = tokenizer.convert_ids_to_tokens(ids)
        if isinstance(tokens, str):
            tokens = [tokens]
        pieces.append((ids, list(tokens)))
    if len(pieces) != len(chars):
        raise ValueError("tokenizer returned a different batch size than requested")
    return pieces


def continuation_label_id(label_id: int, id_to_label: Mapping[int, str], label_to_id: Mapping[str, int]) -> int:
    """Return the label id for a subsequent subtoken of the same character."""

    label = id_to_label[int(label_id)]
    if label.startswith("B-"):
        return int(label_to_id.get("I-" + label[2:], label_id))
    return int(label_id)


def align_text(
    text: str,
    tokenizer: object,
    labels: Sequence[int] | None = None,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    label_to_id: Mapping[str, int] | None = None,
    id_to_label: Mapping[int, str] | None = None,
    pad_to_max_length: bool = True,
) -> AlignedEncoding:
    """Tokenize a KLUE character string while preserving original character indices.

    KoELECTRA's tokenizer drops literal spaces. Passing the literal string
    ``"[unused0]"`` through split-word tokenization is also unsafe because it is
    split into multiple pieces. This function tokenizes only non-space
    characters and inserts the already-resolved ``[unused0]`` id for every
    original space.
    """

    chars = list(text)
    if labels is not None and len(labels) != len(chars):
        raise ValueError(f"labels length {len(labels)} does not match text length {len(chars)}")

    cls_id = _special_id(tokenizer, "cls_token_id", "[CLS]")
    sep_id = _special_id(tokenizer, "sep_token_id", "[SEP]")
    pad_id = _special_id(tokenizer, "pad_token_id", "[PAD]")
    space_id = tokenizer.convert_tokens_to_ids(SPACE_TOKEN)
    if space_id is None or int(space_id) < 0:
        raise ValueError(f"tokenizer vocabulary does not contain {SPACE_TOKEN}")
    space_id = int(space_id)
    unk_id = _special_id(tokenizer, "unk_token_id", "[UNK]")
    if space_id == unk_id:
        raise ValueError(f"tokenizer maps {SPACE_TOKEN} to the unknown-token id")

    input_ids = [cls_id]
    tokens = [getattr(tokenizer, "cls_token", "[CLS]")]
    char_indices: list[int | None] = [None]
    aligned_labels: list[int] | None = [IGNORE_INDEX] if labels is not None else None
    non_space_chars = [char for char in chars if not char.isspace()]
    tokenized_non_space = iter(_batch_tokenize_chars(tokenizer, non_space_chars))

    for char_index, char in enumerate(chars):
        if char.isspace():
            char_token_ids = [space_id]
            char_tokens = [SPACE_TOKEN]
        else:
            char_token_ids, char_tokens = next(tokenized_non_space)

        for piece_index, (token_id, token) in enumerate(zip(char_token_ids, char_tokens)):
            input_ids.append(token_id)
            tokens.append(token)
            char_indices.append(char_index)
            if aligned_labels is not None:
                label_id = int(labels[char_index])
                if piece_index and label_to_id is not None and id_to_label is not None:
                    label_id = continuation_label_id(label_id, id_to_label, label_to_id)
                aligned_labels.append(label_id)

    input_ids.append(sep_id)
    tokens.append(getattr(tokenizer, "sep_token", "[SEP]"))
    char_indices.append(None)
    if aligned_labels is not None:
        aligned_labels.append(IGNORE_INDEX)

    if len(input_ids) > max_length:
        raise ValueError(
            f"encoded length {len(input_ids)} exceeds max_length {max_length}; "
            "increase max_length instead of silently truncating"
        )

    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)

    if pad_to_max_length:
        pad_count = max_length - len(input_ids)
        input_ids.extend([pad_id] * pad_count)
        attention_mask.extend([0] * pad_count)
        token_type_ids.extend([0] * pad_count)
        char_indices.extend([None] * pad_count)
        tokens.extend([getattr(tokenizer, "pad_token", "[PAD]")] * pad_count)
        if aligned_labels is not None:
            aligned_labels.extend([IGNORE_INDEX] * pad_count)

    return AlignedEncoding(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=aligned_labels,
        char_indices=char_indices,
        tokens=tokens,
    )
