import os
import numpy as np
import typer
from typing import Annotated

from monai.data.utils import iter_patch_slices

from cryosiam.data import MrcReader, MrcWriter
from cryosiam.data_structure.dense_simsiam_config import load_dense_simsiam_config


def main(
    config_file_path: Annotated[
        str, typer.Option(help="Path to config file", exists=True)
    ],
):
    cfg = load_dense_simsiam_config(config_file_path)

    writer = MrcWriter(output_dtype=np.float32, overwrite=True)
    writer.set_metadata({"voxel_size": 1})
    reader = MrcReader(read_in_mem=True)

    os.makedirs(os.path.join(cfg.patches_folder, "image"), exist_ok=True)
    files = [x for x in os.listdir(cfg.data_folder) if x.endswith(cfg.file_extension)]
    if cfg.train_files is not None:
        if cfg.val_files is None:
            cfg.val_files = []
        files = [x for x in files if x in cfg.train_files or x in cfg.val_files]

    for file in files:
        print(f"Processing tomo {file}")
        file_path = os.path.join(cfg.data_folder, file)

        image = reader.read(file_path)
        image = image.data
        image.setflags(write=True)

        root_file_name = file.split(cfg.file_extension)[0]

        if len(cfg.parameters.data.patch_size) == 2:
            start_pos = (0, 0)
        else:
            start_pos = (0, 0, 0)

        iter_size = image.shape
        for slices in iter_patch_slices(
            iter_size,
            cfg.parameters.data.patch_size,
            start_pos,
            cfg.parameters.data.patch_overlap,
            padded=False,
        ):
            coords = tuple((coord.start, coord.stop) for coord in slices)
            coords_array = np.asarray(coords)
            patch = image[slices].astype(np.float32)

            if len(cfg.parameters.data.patch_size) == 2:
                y, x = coords_array[0][0], coords_array[1][0]
                patch_file_name = f"{root_file_name}_y{y}_x{x}{cfg.file_extension}"

            else:
                z, y, x = coords_array[0][0], coords_array[1][0], coords_array[2][0]
                patch_file_name = f"{root_file_name}_z{z}_y{y}_x{x}{cfg.file_extension}"

            subtomo_path = os.path.join(cfg.patches_folder, "image", patch_file_name)

            writer.set_data_array(patch, channel_dim=None)
            writer.write(subtomo_path)


if __name__ == "__main__":
    typer.run(main)
