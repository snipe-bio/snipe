"""API module for Snipe."""

class SnipeAPI:
    """A sample API class for managing key-value data."""

    def __init__(self):
        """Initialize the Snipe API with an empty data store."""
        self.data = {}

    def get_data(self, key: str) -> str | None:
        """Retrieve data by key.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            str | None: The value associated with the key, or None if not found.
        """
        return self.data.get(key, None)

    def set_data(self, key: str, value: str) -> None:
        """Set data for a given key.

        Args:
            key (str): The key to set.
            value (str): The value to associate with the key.
        """
        self.data[key] = value


