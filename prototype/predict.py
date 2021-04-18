import os
from typing import List, Dict

import PIL.Image
import pandas as pd

from kidney.datasets.kaggle import KaggleKidneyDatasetReader
from kidney.inference.inference import SlidingWindow
from kidney.utils.checkpoints import CheckpointsStorage, load_experiment, get_factory
from kidney.utils.mask import rle_numba_encode
from prototype.config import configure, PredictConfig


@configure
def main(config: PredictConfig) -> None:
    storage = CheckpointsStorage(config.storage_dir)
    checkpoint = storage.fetch_best_file_in_dir(config.performance_metric)
    factory = get_factory(config.factory_class)
    experiment, meta = load_experiment(factory, checkpoint.path, checkpoint.meta)
    config.sliding_window.transform_input = meta["transformers"].test_preprocessing
    config.sliding_window.transform_output = meta["transformers"].test_postprocessing
    inference = SlidingWindow(
        model=experiment,
        device=config.device,
        debug=config.debug,
        config=config.sliding_window
    )
    predictions = inference.predict_from_reader(
        reader=KaggleKidneyDatasetReader(config.dataset),
        sample_type=config.sample_type,
        encoder=rle_numba_encode if config.encode_masks else None
    )
    save_results(predictions, config.predictions_file, encoded=config.encode_masks)


def save_results(predictions: List[Dict], output_file: str, encoded: bool):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if encoded:
        predictions_df = pd.DataFrame(predictions, columns=["id", "predicted"])
        predictions_df.to_csv(output_file, index=False)
        print("Predictions path:", output_file)
    else:
        os.makedirs(output_file, exist_ok=True)
        for result in predictions:
            image_id = result["id"]
            mask = result["predicted"]
            if mask.max() == 1:
                mask *= 255
            image_file = os.path.join(output_file, f"{image_id}.png")
            PIL.Image.fromarray(mask).save(image_file, format="png")
            print("Mask saved:", image_file)



if __name__ == '__main__':
    main()
