from typing import Any
from lion_core.generic.pile import Pile
from lion_core.generic.component import Component
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.schema import TextNode

import llama_index.core.indices as Indices
from .converter import to_llama_index

NODES_TYPES = TextNode | Component


def create_llama_index(
    nodes: list[NODES_TYPES] | Pile[NODES_TYPES] | NODES_TYPES,
    *args,
    index_type: BaseIndex | str | None = None,
    llm: LLM | None = None,
    return_llama_nodes: bool = False,
    **kwargs: Any,
):
    from llama_index.core import Settings

    if isinstance(nodes, TextNode | Component):
        nodes = [nodes]
    if isinstance(nodes, Pile):
        nodes = list(nodes)

    llama_nodes = [to_llama_index(node) for node in nodes]

    if llm and isinstance(llm, LLM):
        Settings.llm = llm

    if index_type is None:
        index_type = VectorStoreIndex

    elif isinstance(index_type, str):
        index_type: BaseIndex = getattr(Indices, index_type)

    out = index_type(llama_nodes, *args, **kwargs)
    if return_llama_nodes:
        return out, llama_nodes
    return out
