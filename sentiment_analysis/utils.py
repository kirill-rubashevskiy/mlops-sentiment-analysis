from io import StringIO
from pathlib import Path

import dvc.api
import onnxruntime as rt
import pandas as pd
from classifiers import clfs
from preprocessing import preprocessings
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline


def build_model(
    preprocessing_name: str, preprocessing_params: dict, clf_name: str, clf_params
):
    preprocessing = preprocessings[preprocessing_name]
    preprocessing.set_params(**preprocessing_params)
    clf = clfs[clf_name]
    clf.set_params(**clf_params)

    model = Pipeline([(preprocessing_name, preprocessing), (clf_name, clf)])
    return model


def load_data(
    dataset_filename: str, access_key_id: str, secret_access_key: str
) -> pd.DataFrame:
    config = {
        "remote": {
            "yandexcloud": {
                "access_key_id": access_key_id,
                "secret_access_key": secret_access_key,
            },
        },
    }
    contents = dvc.api.read(
        path=f"data/{dataset_filename}",
        repo="https://github.com/kirill-rubashevskiy/mlops-sentiment-analysis/",
        config=config,
    )
    data = pd.read_csv(StringIO(contents))
    return data


def load_model(
    model_filename: str, access_key_id: str, secret_access_key: str
) -> rt.InferenceSession:
    config = {
        "remote": {
            "yandexcloud": {
                "access_key_id": access_key_id,
                "secret_access_key": secret_access_key,
            },
        },
    }
    contents = dvc.api.read(
        path=f"models/{model_filename}",
        repo="https://github.com/kirill-rubashevskiy/mlops-sentiment-analysis/",
        mode="rb",
        config=config,
    )
    sess = rt.InferenceSession(contents, providers=["CPUExecutionProvider"])
    return sess


def save_model(model, model_filename: str, preprocessing_name: str) -> None:
    options = {
        "zipmap": False,
        "extra": {
            model[preprocessing_name].__class__: {
                "separators": [" ", "[.]", "\\?", ",", ";", ":", "\\!", "\\(", "\\)"]
            }
        },
    }

    model_onnx = convert_sklearn(
        model,
        model_filename,
        initial_types=[("input", StringTensorType([None, 1]))],
        options=options,
    )

    with open(f"models/{model_filename}", "wb") as file:
        file.write(model_onnx.SerializeToString())


def save_predictions(predictions: list, predictions_filename: str) -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    Path.mkdir(data_dir / "tmp", exist_ok=True)
    pd.DataFrame(
        {"predict": predictions[0], "predict_proba_1": predictions[1][:, 1]}
    ).to_csv(data_dir / f"tmp/{predictions_filename}", index=False)
