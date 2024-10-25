import numpy as np
from scipy import stats


def compute_statistics(array1, array2):
    # Set the random seed for reproducibility
    # np.random.seed(seed)
    mean1, std1 = np.mean(array1), np.std(array1, ddof=1)
    mean2, std2 = np.mean(array2), np.std(array2, ddof=1)
    n1, n2 = len(array1), len(array2)
    t_stat, p_value = stats.ttest_rel(array1, array2)

    # Calculate pooled standard deviation and Cohen's d
    pooled_sd = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_sd

    return (t_stat, p_value), cohens_d






if __name__ == '__main__':
    # # Example data: Providing raw data arrays
    data1 = np.array([20, 22, 19, 20, 22])
    data2 = np.array([30, 31, 29, 30, 31])
    #
    # Calculate t-test and Cohen's d with raw data
    t_test_result_stats, cohens_d_stats = compute_statistics(array1=data1, array2=data2)


    print("Paired t-test result with summary statistics (t-statistic, p-value):", t_test_result_stats)
    print("Cohen's d with summary statistics:", cohens_d_stats)
