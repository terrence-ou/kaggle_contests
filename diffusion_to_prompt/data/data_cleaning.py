import os
import unicodedata
import pandas as pd

from urllib.request import urlretrieve


def is_english_only(string):
    for s in string:
        cat = unicodedata.category(s)
        if not cat in ['Ll', 'Lu', 'Nd', 'Po', 'Pd', 'Zs']:
            return False
        return True

def get_parquet():
    print("\n--->Getting parquet data...")
    table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
    urlretrieve(table_url, 'metadata.parquet')

def pre_processing():
    print("--->Start processing...")
    df = pd.read_parquet("metadata.parquet", columns=["image_name", "prompt", "width", "height"])
    
    # filter height and width
    df = df[(df["width"] == 512) & (df["height"] == 512)]
    print("\tfile num. after size filtering: ", len(df))
    
    # filter out short and invalid prompts
    df = df[df["prompt"].map(lambda x: len(x.split()) >= 5)]
    df = df[~df["prompt"].str.contains('^(?:\s*|NULL|null|NaN)$', na=True)]
    df = df[df["prompt"].apply(is_english_only)]
    print("\tfile num. after prompt filtering: ", len(df))
    
    # Remove duplicates
    df["head"] = df["prompt"].str[:15]
    df["tail"] = df["prompt"].str[-15:]
    df.drop_duplicates(subset="head", inplace=True)
    df.drop_duplicates(subset="tail", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("\tfile num. after removing duplicates: ", len(df))
    return df

def match_file(df, image_folder):
    print("--->Matching files...")
    images = os.listdir(image_folder)
    df.loc[df["image_name"].isin(images), "filepath"] = image_folder + df["image_name"]
    df = df[["filepath", "prompt"]].copy()
    assert not df["filepath"].isnull().any()
    return df

def df_preprocessing(img_dir):
    get_parquet()
    df = pre_processing()
    df = match_file(df, img_dir)
    df.to_csv("diffusiondf.csv", index=False)


if __name__ == "__main__":
    img_dir = "E:/diffusiondb/dataset/"
    df_preprocessing(img_dir)
    print("Processing finished, file saved as diffusiondf.csv in the same directory.\n")

