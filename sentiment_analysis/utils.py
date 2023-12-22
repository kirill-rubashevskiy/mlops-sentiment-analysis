from io import StringIO
from pathlib import Path

import dvc.api
import onnxruntime as rt
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer


data_dir = Path(__file__).resolve().parent.parent / "data"


def load_data(dataset_filename: str) -> pd.DataFrame:
    contents = dvc.api.read(
        path=f"data/{dataset_filename}",
        repo="https://github.com/kirill-rubashevskiy/mlops-sentiment-analysis/",
    )
    data = pd.read_csv(StringIO(contents))
    return data


def load_model(model_filename: str) -> rt.InferenceSession:
    contents = dvc.api.read(
        path=f"models/{model_filename}",
        repo="https://github.com/kirill-rubashevskiy/mlops-sentiment-analysis/",
        mode="rb",
    )
    sess = rt.InferenceSession(contents, providers=["CPUExecutionProvider"])
    return sess


def save_model(model, model_filename: str) -> None:
    options = {
        "zipmap": False,
        "extra": {
            TfidfVectorizer: {
                "separators": [" ", "[.]", "\\?", ",", ";", ":", "\\!", "\\(", "\\)"]
            }
        },
    }

    model_onnx = convert_sklearn(
        model,
        f'{model_filename.removesuffix(".onnx")}',
        initial_types=[("input", StringTensorType([None, 1]))],
        options=options,
    )

    with open(f"models/{model_filename}", "wb") as file:
        file.write(model_onnx.SerializeToString())


def save_predictions(predictions: list, predictions_filename: str) -> None:
    Path.mkdir(data_dir / "tmp", exist_ok=True)

    pd.DataFrame(
        {"predict": predictions[0], "predict_proba_1": predictions[1][:, 1]}
    ).to_csv(data_dir / f"tmp/{predictions_filename}", index=False)
