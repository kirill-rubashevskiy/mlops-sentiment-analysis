import logging

from utils import load_data, load_model, save_predictions


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # load test data
    data = load_data("data.csv")
    logging.info("Test data loaded")

    data = data.sample(1000)

    features = data["review"].values.reshape((-1, 1))
    # labels = data["sentiment"]

    model = load_model("pipeline_tfidf_logreg.onnx")
    logging.info("Model loaded")

    inputs = {"input": features}
    predictions = model.run(None, inputs)
    logging.info("Model made predictions")

    save_predictions(predictions, "predictions.csv")
    logging.info("Predictions saved")


if __name__ == "__main__":
    main()
