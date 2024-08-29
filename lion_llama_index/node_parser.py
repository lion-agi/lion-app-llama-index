from typing import Any

from lion_core.sys_utils import SysUtil
from llama_index.core.node_parser.interface import NodeParser


def get_llamaindex_node_parser(node_parser: type[NodeParser] | str) -> type[NodeParser]:

    if isinstance(node_parser, str):
        if node_parser == "CodeSplitter":
            SysUtil.check_import("tree_sitter_languages")

        return SysUtil.import_module(
            package_name="llama_index",
            module_name="core.node_parser",
            import_name=node_parser,
        )

    elif issubclass(node_parser, NodeParser):
        return node_parser

    raise TypeError("node_parser must be a string or NodeParser.")

def llamaindex_parse_node(
    documents: list,
    node_parser: Any,
    *args,
    **kwargs,
):
    try:
        parser = get_llamaindex_node_parser(node_parser)
        try:
            parser = parser(*args, **kwargs)
        except Exception:
            parser = parser.from_defaults(*args, **kwargs)
        return parser.get_nodes_from_documents(documents)
    except Exception as e:
        raise ValueError(f"Failed to parse. Error: {e}") from e
