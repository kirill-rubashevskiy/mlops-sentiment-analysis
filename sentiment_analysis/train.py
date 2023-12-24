import logging

import git
import hydra
import mlflow
import numpy as np
from omegaconf import OmegaConf
from sklearn.model_selection import cross_validate
from utils import build_model, load_data, save_model


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: OmegaConf):

    mlflow.set_tracking_uri(uri=cfg.mlflow.uri)

    with mlflow.start_run():

        # load train data
        data = load_data(
            cfg.files.train_data,
            cfg.yandexcloud.access_key_id,
            cfg.yandexcloud.secret_access_key,
        )
        logging.info("Train data loaded")

        data = data.sample(1000)

        features = data["review"]
        labels = data["sentiment"].map({"positive": 1, "negative": 0}).astype(int)

        # create model pipeline
        model = build_model(
            cfg.preprocessing.name, cfg.preprocessing.params, cfg.clf.name, cfg.clf.params
        )
        logging.info("Model pipeline created")

        # log preprocessing and clf params
        preprocessing_params = {
            f"preprocessing__{param}": param_value
            for param, param_value in cfg.preprocessing.params.items()
        }
        clf_params = {
            f"clf__{param}": param_value for param, param_value in cfg.clf.params.items()
        }

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        mlflow.log_params(preprocessing_params)
        mlflow.log_params(clf_params)
        mlflow.log_param("git_commit_id", sha)
        logging.info("Params logged")

        # calculate and log train metrics
        scores = cross_validate(
            model, features, labels, cv=3, scoring=tuple(cfg.evaluation.metrics)
        )
        metrics = {
            f"train_{metric}": np.mean(scores[f"test_{metric}"])
            for metric in cfg.evaluation.metrics
        }
        mlflow.log_metrics(metrics)
        logging.info("Metrics logged")

        # fit model
        model.fit(features, labels)
        logging.info("Model fitted")

        # save fitted model in ONNX format
        save_model(model, cfg.files.model, cfg.preprocessing.name)
        logging.info("Model saved")


if __name__ == "__main__":
    main()
