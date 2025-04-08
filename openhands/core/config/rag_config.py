from pydantic import BaseModel, Field, ValidationError


class RAGConfig(BaseModel):
    top_k: int = Field(default=5)
    key: str = Field(default=None)  #### Set as string for demo purposes
    encoder_inference_endpoint: str = Field(default=None)
    encoder_inference_model: str = Field(default=None)
    reranker_model: str = Field(default=None)
    summarizer_api_key: str = Field(default=None)
    embedding_max_token_size: int = Field(default=8192)
    default_rag_dir: str = Field(default='index_deafault')
    summarizer: str = Field(default=None)
    summarizer_url: str = Field(default=None)
    chroma_db_path: str = Field(default='./data/chroma')

    @classmethod
    def from_toml_section(cls, data: dict) -> dict[str, 'RAGConfig']:
        """
        Create a mapping of RAGConfig instances from a toml dictionary representing the [rag] section.
        """
        # Initialize the result mapping
        rag_mapping: dict[str, RAGConfig] = {}

        try:
            # Create the main RAG config
            rag_mapping['rag'] = cls(
                top_k=data.get('top_k', None),
                key=data.get('key', None),
                encoder_inference_endpoint=data.get('encoder_inference_endpoint', None),
                encoder_inference_model=data.get('encoder_inference_model', None),
                reranker_model=data.get('reranker_model', None),
                summarizer_api_key=data.get('summarizer_api_key', None),
                embedding_max_token_size=data.get('embedding_max_token_size', None),
                default_rag_dir=data.get('default_rag_dir', None),
                summarizer=data.get('summarizer', None),
                summarizer_url=data.get('summarizer_url', None),
            )
        except ValidationError as e:
            raise ValueError(f'Invalid RAG configuration: {e}')

        return rag_mapping
