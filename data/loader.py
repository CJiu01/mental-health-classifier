from datasets import load_dataset
from config import DATASET_NAME, EMOTION_LABELS


def load_emotion_dataset() -> dict:
    """
    Returns train/validation/test splits from dair-ai/emotion.

    Returns
    -------
    {
        "train": {"texts": [...], "labels": [...]},
        "validation": {...},
        "test": {...},
        "label_names": [...]
    }
    """
    dataset = load_dataset(DATASET_NAME)

    def _extract(split):
        return {
            "texts" : list(dataset[split]["text"]),
            "labels": list(dataset[split]["label"]),
        }

    return {
        "train"     : _extract("train"),
        "validation": _extract("validation"),
        "test"      : _extract("test"),
        "label_names": EMOTION_LABELS,
    }


def print_stats(data: dict) -> None:
    from collections import Counter
    for split in ("train", "validation", "test"):
        counts = Counter(data[split]["labels"])
        total  = len(data[split]["labels"])
        print(f"\n[{split.upper()}]  total={total}")
        for idx, name in enumerate(data["label_names"]):
            n = counts[idx]
            print(f"  {name:<10} {n:>5}  ({n/total*100:.1f}%)")


if __name__ == "__main__":
    data = load_emotion_dataset()
    print_stats(data)
    print("\nSample text  :", data["train"]["texts"][0])
    print("Sample label :", data["label_names"][data["train"]["labels"][0]])
