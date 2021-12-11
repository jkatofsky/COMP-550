import pandas as pd 
import argparse

def get_flairs(comments_csvs, flairs_csv):
    dfs = []
    for comment_csv in comments_csvs:
        dfs.append(pd.read_csv(comment_csv))
    df = pd.concat(dfs)
    flairs = pd.Series(sorted(df['flair'].unique()))
    flairs.to_csv(flairs_csv, index=False)

    # df.drop(df.columns.difference(["flair"]), axis=1, inplace=True)
    # df.drop_duplicates(subset="flair", inplace=True)
    # df.sort_values(by="flair", inplace=True)
    # df.to_csv(flairs_csv, header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--comments_csvs', 
        nargs='+',
        type=str,
        required=True
    )
    parser.add_argument(
        '--flairs_csv', 
        type=str,
        default="data/flairs.csv"
    )
    args = parser.parse_args()

    get_flairs(args.comments_csvs, args.flairs_csv)