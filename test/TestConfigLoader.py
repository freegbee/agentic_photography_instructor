import os

from src.utils import SslHelper
from src.utils.ConfigLoader import ConfigLoader


def main():
   config = ConfigLoader().load(env=os.environ["ENV_NAME"])
   print(config)


if __name__ == "__main__":
    SslHelper.create_unverified_ssl_context()
    main()
