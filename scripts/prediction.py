"""Script for inference with the DeepForest model."""

from deepforest import main
import fire

from deepforest_finetuning.prediction import prediction
from deepforest_finetuning.config import PredictionConfig
from deepforest_finetuning.utils import load_config


def prediction_script(config_path: str):
    """Script for inference with the DeepForest model."""

    config = load_config(config_path, PredictionConfig)

    # loading model
    if config.checkpoint_path is not None:
        model = main.load_from_checkpoint(config.checkpoint_path)
    else:
        model = main.deepforest()
        model.use_release()

    prediction(
        model=model,
        image_files=config.image_files,
        predict_tile=config.predict_tile,
        export_config=config.export_config,
        patch_size=config.patch_size,
        patch_overlap=config.patch_overlap,
    )


if __name__ == "__main__":
    fire.Fire(prediction_script)
