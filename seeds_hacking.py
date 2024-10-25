import pandas as pd
import numpy as np
from st_test import compute_statistics


def get_column_as_array(df, column_name):
    """
    Extracts a column from a DataFrame and returns it as a NumPy array.

    Parameters:
    df (pandas.DataFrame): The DataFrame to extract the column from.
    column_name (str): The name of the column to extract.

    Returns:
    numpy.ndarray: The extracted column as a NumPy array.
    """
    if column_name in df.columns:
        return df[column_name].to_numpy()
    else:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")


df = pd.read_excel('randomseeds_results.xlsx', sheet_name='Sheet1')

print(df.columns)


size=1000

pick_size= 10

for col in df.columns:
    print('column: ',col)
    count = 0
    diff_count = 0
    total_cohen_d = 0
    total_t_statistic = 0
    total_p_value = 0
    analysis_set = get_column_as_array(df, col)

    # remove the nan values
    analysis_set = analysis_set[~np.isnan(analysis_set)]

    for i in range(size):
        # random pick 10 from the array

        random_pick_1 = np.random.choice(analysis_set, pick_size)

        # sort the pick and pick the top 3
        sorted_pick_1 = np.sort(random_pick_1)
        top_3 = sorted_pick_1[-3:]
        random_pick_2 = np.random.choice(analysis_set, pick_size)
        low_3 = np.random.choice(analysis_set, 3)

        # calculate the t-test and cohen's d
        t_test_result, cohens_d = compute_statistics(array1=top_3, array2=low_3)


        # check if the t-test is significant
        if t_test_result[1] < 0.05 and np.abs(cohens_d) > 1:
            count += 1

        m1= np.mean(top_3)
        m2= np.mean(low_3)
        std_1= np.std(top_3, ddof=1)
        std_2= np.std(low_3, ddof=1)
        if (m1-std_1)>(m2+std_2):
            diff_count+=1

    print(f"Significant t-test count: {count}/{size}")
    print(f"Significant std test difference count: {diff_count}/{size}")














