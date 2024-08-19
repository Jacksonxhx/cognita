from typing import List
from langchain.embeddings.base import Embeddings
from langchain_milvus import Milvus
from langchain_core.vectorstores import VectorStore
from langchain.docstore.document import Document
from pymilvus import MilvusClient, DataType

from backend.logger import logger
from backend.constants import DEFAULT_BATCH_SIZE_FOR_VECTOR_STORE, DATA_POINT_FQN_METADATA_KEY
from backend.modules.vector_db import BaseVectorDB
from backend.types import DataPointVector, VectorDBConfig

MAX_SCROLL_LIMIT = int(1e6)
BATCH_SIZE = 1000


class MilvusVectorDB(BaseVectorDB):
    def __init__(self, config: VectorDBConfig):
        logger.debug(f"Connecting to milvus using config: {config.model_dump()}")
        if config.local:
            self.uri = config.url if config.url else "./milvus_local.db"
            self.milvus_client = MilvusClient(
                uri=self.uri
            )
        else:
            self.uri = config.url
            self.token = config.api_key
            self.milvus_client = MilvusClient(
                uri=self.uri,
                token=self.token
            )

    def create_collection(self, collection_name: str, embeddings: Embeddings):
        logger.debug(f"[Milvus] Creating new collection {collection_name}")

        # Calculate embedding size
        logger.debug(f"[Milvus] Embedding a dummy doc to get vector dimensions")
        partial_embeddings = embeddings.embed_documents(["Initial document"])
        vector_size = len(partial_embeddings[0])
        logger.debug(f"Vector size: {vector_size}")

        schema = self.milvus_client.create_schema(auto_id=False)
        # todo: confirm the id type and max length
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=vector_size)

        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self.milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            enable_dynamic_field=True
        )
        logger.debug(f"[Milvus] Created new collection {collection_name}")

    # TODO:
    def upsert_documents(self, collection_name: str, documents: List[Document], embeddings: Embeddings,
                         incremental: bool = True):
        if len(documents) == 0:
            logger.warning("No documents to index")
            return
        # get record IDs to be upserted
        logger.debug(
            f"[Qdrant] Adding {len(documents)} documents to collection {collection_name}"
        )
        data_point_fqns = []
        for document in documents:
            if document.metadata.get(DATA_POINT_FQN_METADATA_KEY):
                data_point_fqns.append(
                    document.metadata.get(DATA_POINT_FQN_METADATA_KEY)
                )
        print(data_point_fqns)

        Milvus(
            collection_name=collection_name,
            embedding_function=embeddings,
            connection_args={
                'uri': self.uri,
                'token': self.token,
            }
        ).add_documents(documents=documents)
        logger.debug(
            f"[Milvus] Added {len(documents)} documents to collection {collection_name}"
        )

    def get_collections(self) -> List[str]:
        logger.debug(f"[Milvus] Fetching collections")
        collections = self.milvus_client.list_collections()
        logger.debug(f"[Milvus] Fetched {len(collections)} collections")
        return collections

    def delete_collection(self, collection_name: str):
        logger.debug(f"[Milvus] Deleting {collection_name} collection")
        self.milvus_client.drop_collection(collection_name=collection_name)
        logger.debug(f"[Milvus] Deleted {collection_name} collection")

    def get_vector_store(self, collection_name: str, embeddings: Embeddings) -> VectorStore:
        logger.debug(f"[Milvus] Getting vector store for collection {collection_name}")
        return Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={
                'uri': self.uri,
                'token': self.token,
            }
        )

    def get_vector_client(self):
        logger.debug(f"[Milvus] Getting Milvus client")
        return self.milvus_client

    # TODO
    def list_data_point_vectors(self, collection_name: str, data_source_fqn: str,
                                batch_size: int = DEFAULT_BATCH_SIZE_FOR_VECTOR_STORE) -> List[DataPointVector]:
        pass

    # TODO
    def delete_data_point_vectors(self, collection_name: str, data_point_vectors: List[DataPointVector],
                                  batch_size: int = DEFAULT_BATCH_SIZE_FOR_VECTOR_STORE):
        pass

