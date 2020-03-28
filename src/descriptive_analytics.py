import pandas as pd
import matplotlib.pyplot as plt

from src.constants import *

df = pd.read_excel(DATA_SET_LOCATION)

pd.set_option('display.max_columns', 36)

print('Attribute types: \n' + str(df.dtypes) + '\n')
print('Description of data: \n' + str(df.describe()))

hist = df.hist(bins=10)
plt.subplots_adjust(hspace=0.8)
plt.show()



