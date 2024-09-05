import pandas as pd
import argparse
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency


def load_data(client_interactions_filepath, resources_filepath):
    # load client interactions csv
    ci = pd.read_csv(client_interactions_filepath, dtype={
        'Client ID': 'Int64',
        'County': 'string',
        'Zip': 'Int64',
        'State': 'string',
        'Gender': 'string',
        'Race': 'string',
        'Ethnicity': 'string',
        'Age': 'Int64',
        'Veteran_Status': 'string',
        'Branch_of_Service': 'string',
        'Household_Composition': 'string',
        '#_People_in_Household': 'Int64',
        'Pregnant?': 'string',
        'SDOH_Employment': 'string',
        ' SDOH_Household_Income': 'string',
        'SDOH_Education_Level': 'string',
        'SDOH_Housing_Situation': 'string',
        'SDOH_Utility_Needs': 'string',
        'SDOH_Utility_applied_for_EAP': 'string',
        'SDOH_Food_Insecurity': 'string',
        'SDOH_SNAP_Benefits': 'string',
        'Transportation': 'string',
        'SDOH_Health_Insurance_Coverage': 'string',
        'SDOH_Existing_Health_Coverage_Type': 'string',
        'SDOH_Explore_Healthcare_Options': 'string',
        'SDOH_Interpersonal_Safety': 'string',
        'Client_EditStamp': 'string',
        'Agency_Id': 'Int64',
        'Site_Id': 'Int64',
        'Referral_Site_Name': 'string',
        'Site_Name': 'string',
        'Site_Address_1': 'string',
        'Site_Address_2': 'string',
        'Site_City': 'string',
        'Site_County': 'string',
        'Site_State': 'string',
        'Site_Zip_Code': 'Int64',
        'Referral_Service_ID': 'Int64',
        'InteractionReferral_ReferralsModule_edit_stamp': 'string',
        'Referral_Service_Name': 'string',
        'Referral_Taxonomy_Name': 'string',
        'Referral_Taxonomy_Code': 'string',
        'Referral_Unmet_Need_Reason': 'string',
        'Referral_Taxonomy_Subcategory': 'string'
    }, na_values=['', 'NA', 'N/A', 'NaN'])

    # pandas cannot load time in read_csv, so need to convert it after
    ci['Client_EditStamp'] = pd.to_datetime(ci['Client_EditStamp'])
    ci['InteractionReferral_ReferralsModule_edit_stamp'] = (
        pd.to_datetime(ci['InteractionReferral_ReferralsModule_edit_stamp']))

    # read resources csv
    r = pd.read_csv(resources_filepath)
    return ci, r


def null_analysis(input1, input2):
    df1 = input1
    missing_counts1 = df1.isnull().sum()
    total_values1 = df1.shape[0]
    missing_percentages1 = (missing_counts1 / total_values1) * 100
    output1 = pd.DataFrame({'Missing Count': missing_counts1, 'Missing %': missing_percentages1})

    df2 = input2
    missing_counts2 = df2.isnull().sum()
    total_values2 = df2.shape[0]
    missing_percentages2 = (missing_counts2 / total_values2) * 100
    output2 = pd.DataFrame({'Missing Count': missing_counts2, 'Missing %': missing_percentages2})

    return output1, output2


# Goes from 5623337 total NA values to 209975.
# The remaining are from Site City, County, State, Zip Code, and Referral_Taxonomy_Subcategory.
# As of this point, I (Zach) am unsure if I should remove site info entirely or just drop na values for them.
def clean(input):
    # For the majority of columns, NA means Did not answer.
    # This can happen for multiple reasons, from call emergency to caller refusing to answer.
    output = input

    # In cases where the client did said no or did not answer to the Veteran Status question, it would not be fair to
    # Say "Did not answer" to the Branch of Service question
    output["Veteran_Status"] = output["Veteran_Status"].fillna('Did Not Answer')
    output.loc[output['Veteran_Status'] == 'Did Not Answer', 'Branch_of_Service'] = "Not Applicable"
    output.loc[output['Veteran_Status'] == 'No', 'Branch_of_Service'] = "Not Applicable"

    # Same with those no health insurance being asked what type their health insurance is
    output["SDOH_Health_Insurance_Coverage"] = output["SDOH_Health_Insurance_Coverage"].fillna('Did Not Answer')
    output.loc[output['SDOH_Health_Insurance_Coverage'] == 'Did Not Answer', 'SDOH_Existing_Health_Coverage_Type'] = "Not Applicable"
    output.loc[output['SDOH_Health_Insurance_Coverage'] == 'No', 'SDOH_Existing_Health_Coverage_Type'] = "Not Applicable"

    # Same with those who already have health insurance being asked if they want to explore what options they qualify for
    output.loc[output['SDOH_Health_Insurance_Coverage'] == 'Yes', 'SDOH_Explore_Healthcare_Options'] = "Not Applicable"

    # Same with "pregnant males"
    output["Gender"] = output["Gender"].fillna("Did Not Answer")
    output.loc[output['Gender'] == 'Did Not Answer', 'Pregnant?'] = "Not Applicable"
    output.loc[output['Gender'] == 'Male', 'Pregnant?'] = "Not Applicable"
    output["Pregnant?"] = output["Pregnant?"].fillna("Did Not Answer")

    string_columns_to_fill = ['City', 'County', 'State', 'Gender', 'Race', 'Ethnicity',
                              'Branch_of_Service', 'Household_Composition', 'SDOH_Employment', " SDOH_Household_Income",
                              'SDOH_Education_Level', 'SDOH_Housing_Situation', 'SDOH_Utility_Needs',
                              'SDOH_Utility_applied_for_EAP', 'SDOH_Food_Insecurity', 'SDOH_SNAP_Benefits',
                              'Transportation', 'SDOH_Existing_Health_Coverage_Type',
                              'SDOH_Explore_Healthcare_Options', 'SDOH_Interpersonal_Safety']
    output[string_columns_to_fill] = output[string_columns_to_fill].fillna("Did Not Answer")

    # When the column is of type Integer, we cannot put a string. So, -1 is the default for Did Not Answer, as -1
    # does not come up naturally in the data set
    int_columns_to_fill = ['Zip', 'Age', '#_People_in_Household']
    output[int_columns_to_fill] = output[int_columns_to_fill].fillna(-1)

    # When Referral_Unmet_Need_Reason is empty, it means that the caller has had their need met.
    output['Referral_Unmet_Need_Reason'] = output['Referral_Unmet_Need_Reason'].fillna("Need Met")

    # modify to "Need Unmet" for classification purposes
    output.loc[output['Referral_Unmet_Need_Reason'] != 'Need Met', 'Referral_Unmet_Need_Reason'] = 'Need Unmet'

    # Drop columns that are not used for analysis. Drop Referral_Site_Name because it is a duplicate of Site_Name
    columns_to_drop = ['Client_EditStamp', 'Site_Address_1', 'Site_Address_2', 'Referral_Taxonomy_Code',
                       'Referral_Site_Name']
    output.drop(columns=columns_to_drop, inplace=True)

    # rename SDOH_Household_Income due to typo in column
    output.rename(columns={" SDOH_Household_Income": "SDOH_Household_Income"}, inplace=True)

    # calculate the total number of columns and the count of 'Did Not Answer' values in each row (remember to update constant)
    total_columns = 26
    did_not_answer_count = output.eq('Did Not Answer').sum(axis=1)
    
    # Calculate the percentage of 'Did Not Answer' values in each row
    output['No Response Rate'] = (did_not_answer_count / total_columns) * 100

    return output


# Count and plot the given column with a pie chart
def plot_col_pie(input, col, hide_x = False, hide_y = False, is_ascending=False, exclude=[], top = -1):
    # Count the occurrences of unique values in a column
    val_counts = input[col].value_counts().sort_values(ascending=is_ascending)
    print('Value counts of ' + col)
    print(val_counts)

    # Exclude certain column values (if specified)
    filtered_counts = val_counts[~val_counts.index.isin(exclude)]

    # Only shows top N results (if specified)
    if top != -1:
        filtered_counts = filtered_counts.head(top)

    # Plot the pie chart
    plt.pie(filtered_counts, labels=filtered_counts.index)
    plt.title('Pie Chart of ' + col)
    plt.show()


# Count and plot the given column with a bar chart
def plot_col_bar(input, col, x_label="Value", y_label="Frequency", is_ascending=False, exclude=[], top=-1):
    # Count the occurrences of unique values in a column
    val_counts = input[col].value_counts().sort_values(ascending=is_ascending)
    print('Value counts of ' + col)
    print(val_counts)

    # Exclude certain column values (if specified)
    filtered_counts = val_counts[~val_counts.index.isin(exclude)]

    # Only shows top N results (if specified)
    if top != -1:
        filtered_counts = filtered_counts.head(top)

    # Plot
    plt.bar(filtered_counts.index, filtered_counts)
    plt.xlabel(x_label)
    plt.xticks(rotation='vertical')
    plt.ylabel(y_label)
    plt.title('Bar Chart of ' + col)
    plt.show()


# Perform a correlation test between two variables
def correlation_test_chi_sq(input, x, y):
    # Compute the cross tabulation
    crosstab = pd.crosstab(index=input[x], columns=input[y])

    # Perform the chi-sq test
    res = chi2_contingency(crosstab)

    # Small p value indicates correlation, and vice versa
    print(res[1])


def aggregate_data(input):
    output = input
    # Total number of referrals given to client
    output['Total Referrals'] = output['Referral_Service_ID']
    # Number of unique referrals given to client
    output['Num Unique Referrals'] = output['Referral_Service_ID']
    # Total number of calls client made
    output['Total Calls'] = output['InteractionReferral_ReferralsModule_edit_stamp']

    # Ensure all Referral_Taxonomy_Subcategory are filled with correct subcategory
    ref_tax_subcat = output.groupby(output['Referral_Service_ID']).aggregate({'Referral_Taxonomy_Subcategory': 'first'})
    output = pd.merge(output, ref_tax_subcat, on=['Referral_Service_ID', 'Referral_Taxonomy_Subcategory'], how='right')
    # fill empty subcategories with 'Not Available'
    output['Referral_Taxonomy_Subcategory'] = output['Referral_Taxonomy_Subcategory'].fillna('Not Available')
    
    # Count values for each unique Referral_Taxonomy_Subcategory
    ref_tax_subcat_list = output['Referral_Taxonomy_Subcategory'].unique().tolist()
    ref_tax_subcat_agg = output.groupby('Client ID')['Referral_Taxonomy_Subcategory'].value_counts().unstack(fill_value=0)

    # List of included columns and aggregate function used in group by 
    # (drops Agency_Id, Site_ID, Site_Name, Site_City, Site_County, Site_State, Site_Zip_Code, Referral_Service_ID, 
    #       Referral_Service_Name, Referral_Taxonomy_Name, and Referral_Taxonomy_Subcategory)
    # (adds Total Referrals, Num Unique Referrals, and Total Calls)
    agg_functions = {'Client ID': 'first',
                     'City': 'first',
                     'County': 'first',
                     'Zip': 'first',
                     'State': 'first',
                     'Gender': 'first',
                     'Race': 'first',
                     'Ethnicity': 'first',
                     'Age': 'first',
                     'Veteran_Status': 'first',
                     'Branch_of_Service': 'first',
                     'Household_Composition': 'first',
                     '#_People_in_Household': 'first',
                     'Pregnant?': 'first',
                     'SDOH_Employment': 'first',
                     'SDOH_Household_Income': 'first',
                     'SDOH_Education_Level': 'first',
                     'SDOH_Housing_Situation': 'first',
                     'SDOH_Utility_Needs': 'first',
                     'SDOH_Utility_applied_for_EAP': 'first',
                     'SDOH_Food_Insecurity': 'first',
                     'SDOH_SNAP_Benefits': 'first',
                     'Transportation': 'first',
                     'SDOH_Health_Insurance_Coverage': 'first',
                     'SDOH_Existing_Health_Coverage_Type': 'first',
                     'SDOH_Explore_Healthcare_Options': 'first',
                     'SDOH_Interpersonal_Safety': 'first',
                     'InteractionReferral_ReferralsModule_edit_stamp': 'max',
                     'Referral_Unmet_Need_Reason': 'first',
                     'No Response Rate': 'first',
                     'Total Referrals': 'count',
                     'Num Unique Referrals': 'nunique',
                     'Total Calls': 'nunique'}
    # Apply aggregate functions grouping by Client ID as it is the primary key
    output = output.groupby('Client ID').aggregate(agg_functions)
    # Create columns for each Referral_Taxonomy_Subcatgory and populate with the value counts of each subcategory
    for i in range(len(ref_tax_subcat_list)):
        output["Agg " + ref_tax_subcat_list[i]] = ref_tax_subcat_agg[ref_tax_subcat_list[i]]
    return output


if __name__ == '__main__':
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    # Takes command line arguments
    parser = argparse.ArgumentParser(description="EDA for Indiana 211")
    parser.add_argument('-c', '--client_interactions', required=True, help="filepath to client interaction csv")
    parser.add_argument('-r', '--resources', required=True, help="filepath to the resources csv")
    args = parser.parse_args()
    client_interactions_filepath = args.client_interactions
    resources_filepath = args.resources

    # load the data from user inputted filepaths into pandas dataframes
    ci, r = load_data(client_interactions_filepath, resources_filepath)

    # null1, null2 = null_analysis(ci, r)

    ci2 = clean(ci)
    
    # # plot some cols
    # plot_col_pie(ci2, 'Referral_Unmet_Need_Reason', exclude=['Need Met'])
    # plot_col_bar(ci2, 'County', x_label='County', y_label='Population', top=10)\
    # correlation_test_chi_sq(ci2, 'SDOH_Household_Income','SDOH_Employment')
    # correlation_test_chi_sq(ci2,'SDOH_Household_Income', 'Referral_Unmet_Need_Reason')
    # aggregate data
    ci3 = aggregate_data(ci2)
    ci3.to_csv("aggregated.csv")
