from src.data_loader import (
    load_goemotions,
    get_splits_as_dataframe
)

ds = load_goemotions()

dataframes = get_splits_as_dataframe(ds)

train_df = dataframes["train"]

print(train_df.columns)
print()

print(
    train_df[
        [
            "text",
            "labels_ekman",
            "anger",
            "disgust",
            "fear",
            "joy",
            "sadness",
            "surprise",
            "neutral",
        ]
    ].head(10)
)

label_columns = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral",
]

for i in range(10):
    print("=" * 60)
    print("Text:")
    print(train_df.iloc[i]["text"])

    print("\nlabels_ekman asli:")
    print(train_df.iloc[i]["labels_ekman"])

    print("\nMulti-hot:")
    print(train_df.iloc[i][label_columns].values.astype(int))