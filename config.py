import os
from dotenv import load_dotenv

load_dotenv(override=True)


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


config = Config()


if __name__ == "__main__":
    config = Config()
    print(config.OPENAI_API_KEY)
    print(config.OPENAI_BASE_URL)
    