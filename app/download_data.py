import kaggle
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def main():
    """
    Downloads the car price prediction dataset from Kaggle.
    """
    try:
        LOGGER.info("Downloading dataset from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "hellbuoy/car-price-prediction", path="data/", unzip=True
        )
        LOGGER.info("Dataset downloaded and unzipped successfully in 'data/' folder.")
    except Exception as e:
        LOGGER.error(f"An error occurred: {e}")
        LOGGER.error(
            "Please ensure your Kaggle API credentials (kaggle.json) are correctly "
            "set up. See https://www.kaggle.com/docs/api for more information."
        )

if __name__ == "__main__":
    main()