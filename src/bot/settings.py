from pydantic_settings import BaseSettings
from pydantic import computed_field

from pathlib import Path
import tempfile


class Settings(BaseSettings):
    TELEGRAM_BOT_TOKEN: str
    GROQ_API_KEY: str
    INFERENCE_API_URL: str
    INFERENCE_ACCESS_TOKEN: str
    TMP_DIR: Path = Path(tempfile.gettempdir())

    @computed_field
    def transcribe_status_url(self) -> str:
        return f"{self.INFERENCE_API_URL}/status"

    @computed_field
    def transcribe_url(self) -> str:
        return f"{self.INFERENCE_API_URL}/transcribe"

    @computed_field
    def transcribe_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.INFERENCE_ACCESS_TOKEN}"}
