from typing import Any
from lion_core.libs import strip_lower
from lion_core.sys_utils import SysUtil
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import BaseNode


def get_llama_index_loader(
    reader: type[BasePydanticReader] | str = None,
    *args: Any,
    **kwargs: Any,
) -> BasePydanticReader:
    if reader is None or (isinstance(reader, str) and "simple" in strip_lower(reader)):
        reader = SimpleDirectoryReader

    if not isinstance(reader, str) and not issubclass(reader, BasePydanticReader):
        raise TypeError(f"reader must be a string or a subclass of BasePydanticReader.")

    if isinstance(reader, str):
        package_name, pip_name = parse_reader_name(reader)
        if package_name == "" and pip_name == "":
            raise ValueError(
                f"{reader} is not found. Please directly input llama-index reader class "
                f"or check llama-index documentation for supported readers."
            )
        try:
            reader = SysUtil.check_import(
                package_name=package_name,
                import_name=reader,
                pip_name=pip_name,
            )
            reader = reader()
            loader = reader(*args, **kwargs)
            return loader

        except Exception as e:
            raise AttributeError(
                f"Failed to import/download {reader}, "
                f"please check llama-index documentation to download it "
                f"manually and input the reader object: {e}"
            )

    elif issubclass(reader, BasePydanticReader):
        loader = reader(*args, **kwargs)
        return loader


def parse_reader_name(reader_str):

    package_name = ""
    pip_name = ""

    if reader_str == "PsychicReader":
        package_name = "llama_index.readers.psychic"
        pip_name = "llama-index-readers-psychic"
    elif reader_str == "DeepLakeReader":
        package_name = "llama_index.readers.deeplake"
        pip_name = "llama-index-readers-deeplake"
    elif reader_str == "QdrantReader":
        package_name = "llama_index.readers.qdrant"
        pip_name = "llama-index-readers-qdrant"
    elif reader_str == "DiscordReader":
        package_name = "llama_index.readers.discord"
        pip_name = "llama-index-readers-discord"
    elif reader_str == "SimpleMongoReader":
        package_name = "llama_index.readers.mongodb"
        pip_name = "llama-index-readers-mongodb"
    elif reader_str == "ChromaReader":
        package_name = "llama_index.readers.chroma"
        pip_name = "llama-index-readers-chroma"
    elif reader_str == "MyScaleReader":
        package_name = "llama_index.readers.myscale"
        pip_name = "llama-index-readers-myscale"
    elif reader_str == "FaissReader":
        package_name = "llama_index.readers.faiss"
        pip_name = "llama-index-readers-faiss"
    elif reader_str == "ObsidianReader":
        package_name = "llama_index.readers.obsidian"
        pip_name = "llama-index-readers-obsidian"
    elif reader_str == "SlackReader":
        package_name = "llama_index.readers.slack"
        pip_name = "llama-index-readers-slack"
    elif reader_str == "SimpleWebPageReader":
        package_name = "llama_index.readers.web"
        pip_name = "llama-index-readers-web"
    elif reader_str == "PineconeReader":
        package_name = "llama_index.readers.pinecone"
        pip_name = "llama-index-readers-pinecone"
    elif reader_str == "PathwayReader":
        package_name = "llama_index.readers.pathway"
        pip_name = "llama-index-readers-pathway"
    elif reader_str == "MboxReader":
        package_name = "llama_index.readers.mbox"
        pip_name = "llama-index-readers-mbox"
    elif reader_str == "MilvusReader":
        package_name = "llama_index.readers.milvus"
        pip_name = "llama-index-readers-milvus"
    elif reader_str == "NotionPageReader":
        package_name = "llama_index.readers.notion"
        pip_name = "llama-index-readers-notion"
    elif reader_str == "GithubRepositoryReader":
        package_name = "llama_index.readers.github"
        pip_name = "llama-index-readers-github"
    elif reader_str == "GoogleDocsReader":
        package_name = "llama_index.readers.google"
        pip_name = "llama-index-readers-google"
    elif reader_str == "DatabaseReader":
        package_name = "llama_index.readers.database"
        pip_name = "llama-index-readers-database"
    elif reader_str == "TwitterTweetReader":
        package_name = "llama_index.readers.twitter"
        pip_name = "llama-index-readers-twitter"
    elif reader_str == "WeaviateReader":
        package_name = "llama_index.readers.weaviate"
        pip_name = "llama-index-readers-weaviate"
    elif reader_str == "PandasAIReader":
        package_name = "llama_index.readers.pandas_ai"
        pip_name = "llama-index-readers-pandas-ai"
    elif reader_str == "IntercomReader":
        package_name = "llama_index.readers.intercom"
        pip_name = "llama-index-readers-intercom"

    return package_name, pip_name


def llamaindex_loader(
    reader: str | BasePydanticReader = None,
    /,
    *args: Any,
    reader_args: list = [],
    reader_kwargs: dict = {},
    **kwargs: Any,
) -> list[BaseNode]:
    try:
        loader = get_llama_index_loader(reader=reader, *reader_args, **reader_kwargs)
        documents = loader.load_data(*args, **kwargs)
        return documents
    except Exception as e:
        raise ValueError(f"Failed to read and load data. Error: {e}")
