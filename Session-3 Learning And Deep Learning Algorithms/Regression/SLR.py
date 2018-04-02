import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Feature Vector
sq_ft = np.array([150,200,250,300,350,400,600])
# Dependent Vector/Variable
price = np.array([6450,7450,8450,9450,11450,15450,18450])

slope,intercept,r_value,p_value,stderr =stats.linregress(sq_ft,price)

# Trace the scatter plot for this to visualise
plt.scatter(sq_ft,price,color='red',marker='*')
plt.show()