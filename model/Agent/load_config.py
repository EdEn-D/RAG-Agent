from dotenv import load_dotenv, find_dotenv
import yaml
from pyprojroot import here
import os
import logging

load_dotenv(find_dotenv())


package_name = "Novella_Agent"


class LoadConfig:
    def set_up_logging(self):
        logger = logging.getLogger(package_name)
        return logger

    def __init__(self, path="") -> None:
        logger = self.set_up_logging()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if path:
            config_path = path
        else:
            config_path = os.path.join(current_dir, "..", "configs", "config.yml")

        configs_dir = os.path.dirname(config_path)
        os.makedirs(configs_dir, exist_ok=True)

        try:
            with open(here(config_path)) as cfg:
                config = yaml.load(cfg, Loader=yaml.FullLoader)
                logger.info(f"Config loaded from: {config_path}")
        except Exception as e:
            logger.exception(
                f"Error: {e}, \nconfig.yml not found inside configs folder"
            )
            # break program
            exit()

        self.package_name = package_name

        # Directories
        self.data_directory = config["directories"]["data_directory"]
        self.vectorstore_dir = config["directories"]["vectorstore_dir"]
        self.chat_histrory_dir = config["directories"]["chat_histrory_dir"]

        # Files
        self.queries_path = config["files"]["queries_path"]
        self.guide_path = config["files"]["guide_path"]

        # LLM configurations
        self.llm_smol = config["configs"]["llm_smol"]
        self.llm_large = config["configs"]["llm_large"]
        self.temp_low = config["configs"]["temp_low"]
        self.temp_med = config["configs"]["temp_med"]
        self.temp_high = config["configs"]["temp_high"]
