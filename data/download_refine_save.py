from Datasets import load_dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset argümanlarını alır")

    parser.add_argument(
        "--dataset_hf_id",
        type=str,
        required=True,
        help="Huggingface dataset id"
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "val"],
        required=True,
        help="Which split of the dataset to use"
    )

    parser.add_argument(
        "--columns",
        type=str,
        required=True,
        help="Which columns to select from the dataset, comma-separated"
    )

    return parser.parse_args()


def download_refine_and_save(dataset_hf_id, split, columns):
    # Veri setini indir
    dataset = load_dataset(dataset_hf_id, split=split)

    # Sadece belirtilen kolonları seç
    refined_dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in columns]
    )

    # Refine edilmiş veri setini kaydet
    save_path = f"./refined_{dataset_hf_id.replace('/', '_')}_{split}.parquet"
    refined_dataset.save_to_disk(save_path)
    print(f"Refined dataset saved to {save_path}")

if __name__ == "__main__":
    args = parse_args()

    print(f"Dataset Name: {args.dataset_hf_id}")
    print(f"Split: {args.split}")

    # Kolonları listeye çevir
    columns = [col.strip() for col in args.columns.split(",")]
    print(f"Selected Columns: {columns}")

    download_refine_and_save(args.dataset_hf_id, args.split, columns)
