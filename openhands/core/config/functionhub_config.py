from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError


class FunctionHubConfig(BaseModel):
    """Configuration for Function Hub settings.

    Attributes:
        function_hub_url: URL of the Function Hub.
    """

    function_hub_url: str = Field(default='http://localhost:8000')
    function_hub_wallet_address: str = Field(default='')
    function_hub_api_key: str = Field(default='')

    model_config = {'extra': 'forbid'}

    def validate_function_hub_url(self) -> None:
        """Validate that the function hub URL is valid."""
        if not self.function_hub_url:
            raise ValueError('Function hub URL is required')

        try:
            result = urlparse(self.function_hub_url)
            if not all([result.scheme, result.netloc]):
                raise ValueError(f'Invalid URL format: {self.function_hub_url}')
        except Exception as e:
            raise ValueError(f'Invalid URL {self.function_hub_url}: {str(e)}')

    @classmethod
    def from_toml_section(cls, data: dict) -> dict[str, 'FunctionHubConfig']:
        """
        Create a mapping of FunctionHubConfig instances from a toml dictionary representing the [functionhub] section.

        The configuration is built from all keys in data.

        Returns:
            dict[str, FunctionHubConfig]: A mapping where the key "functionhub" corresponds to the [functionhub] configuration
        """
        # Initialize the result mapping
        functionhub_mapping: dict[str, FunctionHubConfig] = {}

        try:
            functionhub_config = cls(**data)
            functionhub_config.validate_function_hub_url()
            functionhub_mapping['functionhub'] = functionhub_config
        except ValidationError as e:
            raise ValueError(f'Invalid Function Hub configuration: {e}')

        return functionhub_mapping
