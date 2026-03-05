from PIL.Image import Image
from pathlib import Path
from datasets import load_dataset
from tqdm.auto import tqdm


def main() -> None:
    data_root = Path("data")
    out_root: Path = data_root / "cifar10"
    out_root.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("uoft-cs/cifar10")
    label_names: list[str] = dataset["train"].features["label"].names

    for split in dataset.keys():
        split_dir: Path = out_root / str(split)
        split_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(
            tqdm(dataset[split], desc=f"Saving {split}", unit="img")
        ):
            label_name: str = label_names[sample["label"]]
            class_dir: Path = split_dir / label_name
            class_dir.mkdir(parents=True, exist_ok=True)

            image: Image = sample["img"]
            image.save(class_dir / f"{i:06d}.png")


if __name__ == "__main__":
    main()
