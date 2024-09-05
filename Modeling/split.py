from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import argparse
import pandas as pd
from scipy import sparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Test Split")
    parser.add_argument('-c', '--client_interactions', required=True,
                        help="filepath to aggregated and cleaned csv")
    args = parser.parse_args()
    input_path = args.client_interactions
    df1 = pd.read_csv(input_path)

    y = df1["Referral_Unmet_Need_Reason"]
    X = df1.drop(["Referral_Unmet_Need_Reason", "Client ID", "Client ID.1",
                  "InteractionReferral_ReferralsModule_edit_stamp"], axis=1)

    encoder1 = OneHotEncoder()
    encoder2 = LabelEncoder()

    X_encoded = X.apply(encoder2.fit_transform)
    y_encoded = encoder2.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
    
    np.save("features.npy", X_encoded.columns) # Save column names for later use
    sparse.save_npz("X_train.npz", sparse.csr_matrix(X_train))
    sparse.save_npz("X_test.npz", sparse.csr_matrix(X_test))
    pd.DataFrame(y_train).to_csv("y_train.csv")
    pd.DataFrame(y_test).to_csv("y_test.csv")
