from re import T
from typing import Any
from llama_index.core.schema import BaseNode, TextNode
import llama_index.core.schema as Schema
from .utils import LLAMAINDEX_META_FIELDS


def convert_llama_index_to_lion_dict(object_: BaseNode, **kwargs: Any) -> dict:

    dict_ = object_.to_dict()
    dict_["content"] = dict_.pop("text", None)
    metadata = dict_.pop("metadata", {})

    for field in LLAMAINDEX_META_FIELDS:
        metadata[field] = dict_.pop(field, None)

    dict_["metadata"] = {"llama_index_metadata": metadata}

    return dict_


def convert_lion_to_llama_index_dict(subject: dict | Any, **kwargs: Any) -> dict:
    if not isinstance(subject, dict):
        subject = subject.to_dict()

    dict_ = {}
    metadata: dict = subject.pop("metadata", {})
    llama_meta: dict = metadata.pop("llama_index_metadata", {})

    for field in LLAMAINDEX_META_FIELDS:
        dict_[field] = llama_meta.pop(field, None)

    dict_["text"] = subject.pop("content", None)
    dict_["metadata"] = {"lion_metadata": metadata, **llama_meta}

    return dict_


def to_llama_index(
    subject: dict | TextNode | Any,
    convert_kwargs: dict = {},
    llama_node_type: str | BaseNode | None = None,
    **kwargs: Any,
) -> BaseNode:

    if isinstance(subject, BaseNode):
        return subject

    if isinstance(llama_node_type, str):
        llama_node_type: BaseNode = getattr(Schema, llama_node_type)

    if llama_node_type is None:
        llama_node_type = TextNode

    subject = convert_lion_to_llama_index_dict(subject=subject, **convert_kwargs)
    return llama_node_type.from_dict(subject)
