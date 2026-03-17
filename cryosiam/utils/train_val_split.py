import os
import math
import random
from typing import List, Optional, TypeAlias, Union


def supervised_train_val_split(
    data_root,
    seg_root,
    files=None,
    ratio=0.1,
    val_files=None,
    patches_folder=None,
    input_key_name="image",
    output_key_name="labels",
    file_ext=".mrc",
):
    train_data = []
    val_data = []
    data_files = os.listdir(data_root) if files is None else files
    if val_files is None:
        random.shuffle(data_files)
        ratio_ind = math.ceil(len(data_files) * ratio)
        train_files = data_files[:-ratio_ind]
        val_files = data_files[-ratio_ind:]
    else:
        train_files = data_files
        val_files = val_files
    if patches_folder is not None:
        patches_files = [
            os.path.basename(x)
            for x in os.listdir(os.path.join(patches_folder, input_key_name))
        ]

    for idx, file in enumerate(train_files):
        if patches_folder is not None:
            file_prefix = file.split(file_ext)[0]
            patch_files = [x for x in patches_files if x.startswith(file_prefix + "_")]
            for patch_file in patch_files:
                patch_file = os.path.basename(patch_file)
                train_data.append(
                    {
                        input_key_name: os.path.join(
                            patches_folder, input_key_name, patch_file
                        ),
                        output_key_name: os.path.join(
                            patches_folder, output_key_name, patch_file
                        ),
                    }
                )
        else:
            train_data.append(
                {
                    input_key_name: os.path.join(data_root, file),
                    output_key_name: os.path.join(seg_root, file),
                }
            )

    for idx, file in enumerate(val_files):
        if patches_folder is not None:
            file_prefix = file.split(file_ext)[0]
            patch_files = [x for x in patches_files if x.startswith(file_prefix + "_")]
            for patch_file in patch_files:
                patch_file = os.path.basename(patch_file)
                val_data.append(
                    {
                        input_key_name: os.path.join(
                            patches_folder, input_key_name, patch_file
                        ),
                        output_key_name: os.path.join(
                            patches_folder, output_key_name, patch_file
                        ),
                    }
                )
        else:
            val_data.append(
                {
                    input_key_name: os.path.join(data_root, file),
                    output_key_name: os.path.join(seg_root, file),
                }
            )

    return train_data, val_data


def supervised_instance_train_val_split(
    data_root,
    out_root,
    noisy_data_root=None,
    files=None,
    ratio=0.1,
    val_files=None,
    patches_folder=None,
    file_ext=".mrc",
):
    train_data = []
    val_data = []
    data_files = os.listdir(data_root) if files is None else files
    if val_files is None:
        random.shuffle(data_files)
        ratio_ind = math.ceil(len(data_files) * ratio)
        train_files = data_files[:-ratio_ind]
        val_files = data_files[-ratio_ind:]
    else:
        train_files = data_files
        val_files = val_files
    if patches_folder is not None:
        patches_files = [
            os.path.basename(x)
            for x in os.listdir(os.path.join(patches_folder, "image"))
        ]

    for idx, file in enumerate(train_files):
        if patches_folder is not None:
            file_prefix = file.split(file_ext)[0]
            patch_files = [x for x in patches_files if x.startswith(file_prefix + "_")]
            for patch_file in patch_files:
                patch_file = os.path.basename(patch_file)
                if noisy_data_root:
                    train_data.append(
                        {
                            "image": os.path.join(patches_folder, "image", patch_file),
                            "noisy_image": os.path.join(
                                patches_folder, "noisy_image", patch_file
                            ),
                            "foreground": os.path.join(
                                patches_folder, "foreground", patch_file
                            ),
                            "distances": os.path.join(
                                patches_folder, "distances", patch_file
                            ),
                            "boundaries": os.path.join(
                                patches_folder, "boundaries", patch_file
                            ),
                        }
                    )
                else:
                    train_data.append(
                        {
                            "image": os.path.join(patches_folder, "image", patch_file),
                            "foreground": os.path.join(
                                patches_folder, "foreground", patch_file
                            ),
                            "distances": os.path.join(
                                patches_folder, "distances", patch_file
                            ),
                            "boundaries": os.path.join(
                                patches_folder, "boundaries", patch_file
                            ),
                        }
                    )
        else:
            if noisy_data_root:
                train_data.append(
                    {
                        "image": os.path.join(data_root, file),
                        "noisy_image": os.path.join(noisy_data_root, file),
                        "foreground": os.path.join(out_root, "foreground", file),
                        "distances": os.path.join(out_root, "distances", file),
                        "boundaries": os.path.join(out_root, "boundaries", file),
                    }
                )
            else:
                train_data.append(
                    {
                        "image": os.path.join(data_root, file),
                        "foreground": os.path.join(out_root, "foreground", file),
                        "distances": os.path.join(out_root, "distances", file),
                        "boundaries": os.path.join(out_root, "boundaries", file),
                    }
                )

    for idx, file in enumerate(val_files):
        if patches_folder is not None:
            file_prefix = file.split(file_ext)[0]
            patch_files = [x for x in patches_files if x.startswith(file_prefix + "_")]
            for patch_file in patch_files:
                patch_file = os.path.basename(patch_file)
                if noisy_data_root:
                    val_data.append(
                        {
                            "image": os.path.join(patches_folder, "image", patch_file),
                            "noisy_image": os.path.join(
                                patches_folder, "noisy_image", patch_file
                            ),
                            "foreground": os.path.join(
                                patches_folder, "foreground", patch_file
                            ),
                            "distances": os.path.join(
                                patches_folder, "distances", patch_file
                            ),
                            "boundaries": os.path.join(
                                patches_folder, "boundaries", patch_file
                            ),
                        }
                    )
                else:
                    val_data.append(
                        {
                            "image": os.path.join(patches_folder, "image", patch_file),
                            "foreground": os.path.join(
                                patches_folder, "foreground", patch_file
                            ),
                            "distances": os.path.join(
                                patches_folder, "distances", patch_file
                            ),
                            "boundaries": os.path.join(
                                patches_folder, "boundaries", patch_file
                            ),
                        }
                    )
        else:
            if noisy_data_root:
                val_data.append(
                    {
                        "image": os.path.join(data_root, file),
                        "noisy_image": os.path.join(noisy_data_root, file),
                        "foreground": os.path.join(out_root, "foreground", file),
                        "distances": os.path.join(out_root, "distances", file),
                        "boundaries": os.path.join(out_root, "boundaries", file),
                    }
                )
            else:
                val_data.append(
                    {
                        "image": os.path.join(data_root, file),
                        "foreground": os.path.join(out_root, "foreground", file),
                        "distances": os.path.join(out_root, "distances", file),
                        "boundaries": os.path.join(out_root, "boundaries", file),
                    }
                )

    return train_data, val_data


def supervised_semantic_train_val_split(
    data_root,
    seg_root,
    out_root,
    noisy_data_root=None,
    files=None,
    ratio=0.1,
    val_files=None,
    patches_folder=None,
    file_ext=".mrc",
):
    train_data = []
    val_data = []
    data_files = os.listdir(data_root) if files is None else files
    if val_files is None:
        random.shuffle(data_files)
        ratio_ind = math.ceil(len(data_files) * ratio)
        train_files = data_files[:-ratio_ind]
        val_files = data_files[-ratio_ind:]
    else:
        train_files = data_files
        val_files = val_files
    if patches_folder is not None:
        patches_files = [
            os.path.basename(x)
            for x in os.listdir(os.path.join(patches_folder, "image"))
        ]

    for idx, file in enumerate(train_files):
        if patches_folder is not None:
            file_prefix = file.split(file_ext)[0]
            patch_files = [x for x in patches_files if x.startswith(file_prefix + "_")]
            for patch_file in patch_files:
                patch_file = os.path.basename(patch_file)
                if noisy_data_root:
                    train_data.append(
                        {
                            "image": os.path.join(patches_folder, "image", patch_file),
                            "noisy_image": os.path.join(
                                patches_folder, "noisy_image", patch_file
                            ),
                            "labels": os.path.join(
                                patches_folder, "labels", patch_file
                            ),
                            "distances": os.path.join(
                                patches_folder,
                                "distances",
                                f"{patch_file.split(file_ext)[0]}.npz",
                            ),
                        }
                    )
                else:
                    train_data.append(
                        {
                            "image": os.path.join(patches_folder, "image", patch_file),
                            "labels": os.path.join(
                                patches_folder, "labels", patch_file
                            ),
                            "distances": os.path.join(
                                patches_folder,
                                "distances",
                                f"{patch_file.split(file_ext)[0]}.npz",
                            ),
                        }
                    )
        else:
            if noisy_data_root:
                train_data.append(
                    {
                        "image": os.path.join(data_root, file),
                        "noisy_image": os.path.join(noisy_data_root, file),
                        "labels": os.path.join(seg_root, file),
                        "distances": os.path.join(
                            out_root, "distances", f"{file.split(file_ext)[0]}.npz"
                        ),
                    }
                )
            else:
                train_data.append(
                    {
                        "image": os.path.join(data_root, file),
                        "labels": os.path.join(seg_root, file),
                        "distances": os.path.join(
                            out_root, "distances", f"{file.split(file_ext)[0]}.npz"
                        ),
                    }
                )

    for idx, file in enumerate(val_files):
        if patches_folder is not None:
            file_prefix = file.split(file_ext)[0]
            patch_files = [x for x in patches_files if x.startswith(file_prefix + "_")]
            for patch_file in patch_files:
                patch_file = os.path.basename(patch_file)
                if noisy_data_root:
                    val_data.append(
                        {
                            "image": os.path.join(patches_folder, "image", patch_file),
                            "noisy_image": os.path.join(
                                patches_folder, "noisy_image", patch_file
                            ),
                            "labels": os.path.join(
                                patches_folder, "labels", patch_file
                            ),
                            "distances": os.path.join(
                                patches_folder,
                                "distances",
                                f"{patch_file.split(file_ext)[0]}.npz",
                            ),
                        }
                    )
                else:
                    val_data.append(
                        {
                            "image": os.path.join(patches_folder, "image", patch_file),
                            "labels": os.path.join(
                                patches_folder, "labels", patch_file
                            ),
                            "distances": os.path.join(
                                patches_folder,
                                "distances",
                                f"{patch_file.split(file_ext)[0]}.npz",
                            ),
                        }
                    )
        else:
            if noisy_data_root:
                val_data.append(
                    {
                        "image": os.path.join(data_root, file),
                        "noisy_image": os.path.join(noisy_data_root, file),
                        "labels": os.path.join(seg_root, file),
                        "distances": os.path.join(
                            out_root, "distances", f"{file.split(file_ext)[0]}.npz"
                        ),
                    }
                )
            else:
                val_data.append(
                    {
                        "image": os.path.join(data_root, file),
                        "labels": os.path.join(seg_root, file),
                        "distances": os.path.join(
                            out_root, "distances", f"{file.split(file_ext)[0]}.npz"
                        ),
                    }
                )

    return train_data, val_data


StrPath: TypeAlias = Union[str, os.PathLike[str]]


def basic_train_val_split(
    data_root: StrPath,
    noisy_data_root: Optional[StrPath],
    files: Optional[List[str]],
    ratio: float,
    patches_folder: Optional[StrPath],
    noisy_patches_folder: Optional[StrPath],
    file_ext: str = ".mrc",
    val_files: Optional[List[str]] = None,
):
    train_data = []
    val_data = []
    data_files = os.listdir(data_root) if files is None else files
    if val_files is None:
        random.shuffle(data_files)
        ratio_ind = math.ceil(len(data_files) * ratio)
        train_files = data_files[:-ratio_ind]
        val_files = data_files[-ratio_ind:]
    else:
        train_files = data_files
        val_files = val_files
    if patches_folder is not None:
        patches_files = [
            os.path.basename(x)
            for x in os.listdir(os.path.join(patches_folder, "image"))
        ]

    for _, file in enumerate(train_files):
        if patches_folder is not None:
            file_prefix = file.split(file_ext)[0]
            patch_files = [x for x in patches_files if x.startswith(file_prefix + "_")]
            for patch_file in patch_files:
                patch_file = os.path.basename(patch_file)
                if noisy_data_root:
                    assert (
                        noisy_patches_folder is not None
                    ), "noisy_patches_folder must be provided if noisy_data_root is provided"
                    train_data.append(
                        {
                            "image": os.path.join(patches_folder, "image", patch_file),
                            "noisy_image": os.path.join(
                                noisy_patches_folder, "noisy_image", patch_file
                            ),
                        }
                    )
                else:
                    train_data.append(
                        {
                            "image": os.path.join(patches_folder, "image", patch_file),
                        }
                    )
        else:
            if noisy_data_root:
                train_data.append(
                    {
                        "image": os.path.join(data_root, file),
                        "noisy_image": os.path.join(noisy_data_root, file),
                    }
                )
            else:
                train_data.append(
                    {
                        "image": os.path.join(data_root, file),
                    }
                )
    for _, file in enumerate(val_files):
        if patches_folder is not None:
            file_prefix = file.split(file_ext)[0]
            patch_files = [x for x in patches_files if x.startswith(file_prefix + "_")]
            for patch_file in patch_files:
                patch_file = os.path.basename(patch_file)
                if noisy_data_root:
                    assert (
                        noisy_patches_folder is not None
                    ), "noisy_patches_folder must be provided if noisy_data_root is provided"
                    val_data.append(
                        {
                            "image": os.path.join(patches_folder, "image", patch_file),
                            "noisy_image": os.path.join(
                                noisy_patches_folder, "noisy_image", patch_file
                            ),
                        }
                    )
                else:
                    val_data.append(
                        {
                            "image": os.path.join(patches_folder, "image", patch_file),
                        }
                    )
        else:
            if noisy_data_root:
                val_data.append(
                    {
                        "image": os.path.join(data_root, file),
                        "noisy_image": os.path.join(noisy_data_root, file),
                    }
                )
            else:
                val_data.append(
                    {
                        "image": os.path.join(data_root, file),
                    }
                )

    return train_data, val_data
