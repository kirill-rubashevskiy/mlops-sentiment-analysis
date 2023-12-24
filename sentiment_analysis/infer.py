import logging

import hydra
from omegaconf import OmegaConf
from utils import load_data, load_model, save_predictions


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: OmegaConf):
    # load test data
    data = load_data(
        cfg.files.test_data,
        cfg.yandexcloud.access_key_id,
        cfg.yandexcloud.secret_access_key,
    )
    logging.info("Test data loaded")

    data = data.sample(1000)

    features = data["review"].values.reshape((-1, 1))
    # labels = data["sentiment"]

    model = load_model(
        cfg.files.model, cfg.yandexcloud.access_key_id, cfg.yandexcloud.secret_access_key
    )
    logging.info("Model loaded")

    inputs = {"input": features}
    predictions = model.run(None, inputs)
    logging.info("Model made predictions")

    save_predictions(predictions, "predictions.csv")
    logging.info("Predictions saved")


if __name__ == "__main__":
    main()
