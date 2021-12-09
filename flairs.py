import pandas as pd 
import argparse

def get_flairs(comments_csv,flairs_csv):
    df = pd.read_csv(comments_csv)
    flairs = df["flair"].to_list()
    flairs = list(set(flairs))
    flairs.sort()
    flairs = pd.DataFrame(flairs)
    flairs.to_csv(flairs_csv, header=False, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--comments_csv', 
        type=str,
        default="data/1000-new-posts-all-comments-only-flairs.csv"
    )
    parser.add_argument(
        '--flairs_csv', 
        type=str,
        default="data/flairs.csv"
    )
    args = parser.parse_args()

    get_flairs(args.comments_csv, args.flairs_csv)