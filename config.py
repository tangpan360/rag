import os
from dotenv import load_dotenv

load_dotenv(override=True)


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
    BAILIAN_BASE_URL = os.getenv("BAILIAN_BASE_URL")
    CLOUD_API_KEY = os.getenv("CLOUD_API_KEY")
    CLOUD_BASE_URL = os.getenv("CLOUD_BASE_URL")


config = Config()


if __name__ == "__main__":
    config = Config()
    print(config.OPENAI_API_KEY)
    print(config.OPENAI_BASE_URL)
    print(config.BAILIAN_API_KEY)
    print(config.BAILIAN_BASE_URL)
    print(config.CLOUD_API_KEY)
    print(config.CLOUD_BASE_URL)
    