import pandas as pd
import argparse


def repeats(df):
    non_uniques = df[df.duplicated(subset=['Client ID'], keep=False)]
    non_unique_rows_sorted = non_uniques.sort_values(by='Client ID')
    print(non_unique_rows_sorted[non_unique_rows_sorted['Client ID'] == 172])
    return


if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    # Takes command line arguments
    parser = argparse.ArgumentParser(description="EDA for Indiana 211")
    parser.add_argument('-c', '--client_interactions', required=True, help="filepath to cleaned client interaction csv")
    args = parser.parse_args()
    client_interactions_filepath = args.client_interactions

    repeats(pd.read_csv(client_interactions_filepath))



