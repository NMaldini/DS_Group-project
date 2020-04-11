import pandas as pd

from src.constants import *
from src.plotter import *

df = pd.read_excel(DATA_SET_LOCATION)

pd.set_option('display.max_columns', 36)


def get_data_dict(data_dict_xlsx_file, sheet):
    return pd.read_excel(data_dict_xlsx_file, sheet_name=sheet)


# Inspection of values
data_dict_df = get_data_dict(DATA_DICT_LOCATION, DATA_DICT_SHEET)


def clean_up_step_1(df, data_dict_df):
    # Summary stats of attributes
    print('Attribute types: \n' + str(df.dtypes) + '\n')
    print('Description of data: \n' + str(df.describe()))
    # Identify missing, invalid, out of range values
    cn_data_columns = list(data_dict_df.loc[data_dict_df['Data Type'] == DATA_TYPES[0]]['Attribute Name'])
    md_data_columns = list(data_dict_df.loc[data_dict_df['Data Type'] == DATA_TYPES[1]]['Attribute Name'])
    mc_data_columns = list(data_dict_df.loc[data_dict_df['Data Type'] == DATA_TYPES[2]]['Attribute Name'])
    #
    # 4/38 attributes here
    cn_data = df[cn_data_columns]

    # Find invalid and missing values
    for col in cn_data:
        print('Unique values of attribute - {}: '.format(str(col)))
        print(cn_data[col].unique())

    # 11/38 attributes
    md_data = df[md_data_columns]
    # Missing values
    print(md_data.isna().any())

    # invalid, out of range values
    hist = md_data.hist(bins=10)
    plt.subplots_adjust(hspace=0.8)
    plt.show()

    # 23/ 38 attributes
    mc_data = df[mc_data_columns]
    # Missing values
    print(mc_data.isna().any())
    # invalid, out of range values
    hist = mc_data.hist(bins=10)
    plt.subplots_adjust(hspace=0.8)
    plt.show()


def rev_by_district(df: pd.DataFrame):
    # Pie chart
    rev_by_disctrict = df.groupby('district').agg({'revenue_rs': 'sum'}).sort_values('revenue_rs',
                                                                                     ascending=False).reset_index()

    # the top 5
    df2 = rev_by_disctrict[:7].copy()

    # others
    new_row = pd.DataFrame(data={
        'district': ['others'],
        'revenue_rs': [rev_by_disctrict['revenue_rs'][7:].sum()]
    })

    # combining top 5 with others
    df2 = pd.concat([df2, new_row])

    labels = df2.district
    sizes = df2.revenue_rs
    plot_py_chart(sizes, labels, 'Revenue by district')


# plot_py_chart(df)

def district_avgs(df: pd.DataFrame):
    disctrict_avgs = df.groupby('district').mean().sort_values('revenue_rs', ascending=False).reset_index()

    df2 = disctrict_avgs[['district', 'revenue_rs']][:7].copy()

    # others
    new_row = pd.DataFrame(data={
        'district': ['others'],
        'revenue_rs': [disctrict_avgs['revenue_rs'][7:].mean()]
    })

    # combining top 5 with others
    df2 = pd.concat([df2, new_row])

    labels = df2.district
    sizes = df2.revenue_rs
    plot_py_chart(sizes, labels, 'Average revenue per person by district')
    x = 5


def n_sim_total(df: pd.DataFrame):
    d_sim_grp = df.groupby('smart_ph_flag').size().reset_index()

    labels = ['Non Smart Phone', 'Smart Phone']
    sizes = d_sim_grp[0]
    plot_py_chart(sizes, labels, 'Customer count by Smart Phone Flag')
    x = 5


# district_avgs(df)

def n_sim_and_rev_1(df: pd.DataFrame):
    sel_cols_df = df[['dual_sim_flag', 'revenue_rs']]
    colors = (0, 0, 0)
    area = np.pi * 3

    # Plot
    plt.scatter(df['dual_sim_flag'], df['revenue_rs'], s=area, c=colors, alpha=0.5)
    plt.title('Scatter plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def n_sim_and_rev_2(df: pd.DataFrame):
    n_sim_avgs = df.groupby('dual_sim_flag').mean().sort_values('revenue_rs', ascending=False).reset_index()
    print(n_sim_avgs[['dual_sim_flag', 'revenue_rs']])
    objects = n_sim_avgs['dual_sim_flag']
    y_pos = np.arange(len(objects))
    performance = n_sim_avgs['revenue_rs']

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel('Usage')
    plt.title('Programming language usage')

    plt.show()


def avg_rev_and_distr(df: pd.DataFrame):
    disctrict_avgs = df.groupby('district').agg({'revenue_rs': 'mean', 'dual_sim_flag': 'size'}) \
        .rename(columns={'revenue_rs': 'avg_rev', 'dual_sim_flag': 'customer_count'}) \
        .sort_values('avg_rev', ascending=False).reset_index()

    plot_bar(disctrict_avgs['district'], disctrict_avgs['avg_rev'], disctrict_avgs['customer_count'], '', 'title')

# seaborn_scatter(df['data_mb'], df['revenue_rs'], 'Data Usage vs. Revenue', 'Data usage (MB)', 'Revenue (Rs.)')
# seaborn_scatter(df['total_og_min'], df['revenue_rs'], 'Out going call duration vs. Revenue', 'Outgoing call duration (minutes)', 'Revenue (Rs.)')
# seaborn_scatter(df['total_og_min'], df['data_mb'], 'Out going call duration vs. Data Usage', 'Outgoing call duration (minutes)', 'Revenue (Rs.)')
# buble_col(df['total_og_min'], df['data_mb'], df['revenue_rs'], df['time_since_last_activity'], 'Out going call duration, Data Usage and Revenue', 'total_og_min', 'data_mb')
# seaborn_scatter(df['time_since_last_recharge'], df['revenue_rs'], 'Time since last recharge vs. Revenue', 'Time since last recharge (days)', 'Revenue (Rs.)')
