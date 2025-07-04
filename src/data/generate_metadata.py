import pandas as pd
import os

"""
Script per generare il file metadata.csv a partire dal CSV di sentiment:
- legge data/processed/video_sentiment_data_0.65.csv
- mantiene solo video_name e category (rinominata emotion)
- salva in data/processed/metadata.csv
"""


def main():
    input_path = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "data",
        "processed",
        "video_sentiment_data_0.65.csv",
    )
    df = pd.read_csv(input_path)
    # Mantieni solo colonne video_name e category -> emotion
    df = df[["video_name", "category"]].rename(columns={"category": "emotion"})

    output_dir = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, "data", "processed"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(output_path, index=False)
    print(f"Metadata salvato in {output_path}")


if __name__ == "__main__":
    main()
