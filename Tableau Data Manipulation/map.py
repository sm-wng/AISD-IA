import pandas as pd
import geopandas as gpd
import argparse
from pandasql import sqldf


def load_json(filepath):
    gdf = gpd.read_file(filepath)
    gdf = gdf.drop(['STATEFP10', 'GEOID10', 'CLASSFP10', 'MTFCC10',
                    'ALAND10', 'AWATER10', 'PARTFLG10', 'FUNCSTAT10'], axis=1)
    gdf["ZCTA5CE10"] = gdf["ZCTA5CE10"].astype(int)
    return gdf


def load_indiana(filepath):
    df = pd.read_csv(filepath)[["Zip", "Referral_Unmet_Need_Reason"]]
    dummies = pd.get_dummies(df['Referral_Unmet_Need_Reason'])

    # Renaming columns to match the requested format
    dummies.columns = ['Need Met', 'Need Unmet']
    df = pd.concat([df, dummies], axis=1).drop(["Referral_Unmet_Need_Reason"], axis=1)

    grouped_df = df.groupby('Zip').sum().reset_index()

    # Calculating the "Percent Unmet" column
    grouped_df['Percent Unmet'] = grouped_df['Need Unmet'] / (grouped_df['Need Met'] + grouped_df['Need Unmet']) * 100

    return grouped_df


def merge(data, geometry):
    output = pd.merge(data, geometry, left_on="Zip", right_on="ZCTA5CE10", how="inner").drop(["ZCTA5CE10"], axis=1)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gets data ready for geographical Tableau Map")
    parser.add_argument('-i', '--indiana211', required=True, help="AGGREGATED Indiana 211 Data")
    parser.add_argument('-j', '--json', required=True, help="geojson of Indiana Zip codes")
    args = parser.parse_args()
    json_path = args.json
    indiana_path = args.indiana211
    json = load_json(json_path)
    indiana = load_indiana(indiana_path)
    merged = merge(indiana, json)
    print(merged)

