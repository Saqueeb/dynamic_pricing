import numpy as np
import pandas as pd
from pricing_ml import*
dataset = pd.read_csv(r'C:\Users\Zia.Accelx\Desktop\Pricing_ML\Final Codes\one_month_observation.csv')

test = {'0': [4], '1': [0], '2': [0], '3': [1], '4': [5]}      #weights of which need prediction
test = pd.DataFrame(test)
result = get_result(dataset, test)
price = int(result[0])
test = rename_cols(test, price)
plt.plot(test.iloc[0])
plt.title('Suggested Price: {}'.format(price))
plt.show()
df1 = better_params(test)
better_result = get_result(dataset, df1)
better_price = int(better_result[0])
df1 = rename_cols(df1, better_price)
plt.plot(df1.iloc[0])
plt.title('Suggested Price: {}'.format(better_price))
plt.show()