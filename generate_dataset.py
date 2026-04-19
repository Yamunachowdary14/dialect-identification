import pandas as pd
import random

df = pd.read_csv("dialect_dataset.csv")

prefixes = ["inka", "asalu", "aithe", "inka mari", ""]
suffixes = ["ippudu", "ra", "inka", "", "kada"]
extras = ["em jarugutundi", "emi scene", "em matter", "em situation"]

new_data = []

for _, row in df.iterrows():
    text = row["text"]
    label = row["label"]

    for _ in range(15):   # 🔥 10 rows → 150 rows per region approx
        t = text

        # add prefix
        if random.random() > 0.3:
            t = random.choice(prefixes) + " " + t

        # replace words
        if "em" in t and random.random() > 0.5:
            t = t.replace("em", random.choice(extras))

        # add suffix
        if random.random() > 0.5:
            t = t + " " + random.choice(suffixes)

        # small variations
        t = t.replace("nuv", random.choice(["nuv", "nuvvu"]))

        new_data.append([t.strip(), label])

# create dataframe
new_df = pd.DataFrame(new_data, columns=["text", "label"])

# remove duplicates
new_df = new_df.drop_duplicates()

# save
new_df.to_csv("dialect_dataset_large.csv", index=False)

print("✅ Final dataset size:", len(new_df))