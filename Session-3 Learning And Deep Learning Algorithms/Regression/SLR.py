import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Feature Vector
sq_ft = np.array([150,200,250,300,350,400,600])
# Dependent Vector/Variable
price = np.array([55000000,95000000,12000000,16000000,27000000,30000000,57000000])

slope,intercept,r_value,p_value,stderr =stats.linregress(sq_ft,price)

# Trace the scatter plot for this to visualise
plt.scatter(sq_ft,price,color='red',marker='*')
plt.show()

# Trace the scatter plot and pass a line through these points so to visualise the accuracy of our model
plt.scatter(sq_ft,price,color='blue',marker='o')
plt.plot(sq_ft,slope*sq_ft+intercept,color='black')
plt.show()

print(r_value**2)

new_x = 550
new_y = new_x*slope + intercept
print(new_y)

# The property price for a random area
# where the square feet is 550 is 36 crores around
# using linear regression

# Clearly SLR sucks for such datasets