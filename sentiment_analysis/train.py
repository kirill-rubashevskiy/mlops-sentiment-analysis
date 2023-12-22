import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils import load_data, save_model


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # load train data
    data = load_data("data.csv")
    logging.info("Train data loaded")

    data = data.sample(1000)

    features = data["review"]
    labels = data["sentiment"]

    # create model pipeline
    model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])

    # fit model
    model.fit(features, labels)
    logging.info("Model fitted")

    # save fitted model in ONNX format
    save_model(model, "pipeline_tfidf_logreg.onnx")
    logging.info("Model saved")


if __name__ == "__main__":
    main()
