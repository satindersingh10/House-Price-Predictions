# Predicting House Prices Using Multi-Linear Regrssion

# Importing Libraries
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Gather Data
boston_dataset = load_boston()
data = pd.DataFrame(data = boston_dataset.data, columns = boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis = 1)
log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns = ['PRICE'])

#property_stats = np.ndarray(shape=(1,11))
RM_IDX = 5
PTRATIO_IDX = 10
CHAS_IDX = 3
#property_stats[0][0] = 0.02
property_stats = features.mean().values.reshape(1,11)

# Creating regression model
regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

# Calculating MSE and RMSE
MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)

def get_log_estimate(nr_rooms,
                     students_per_classroom,
                     next_to_river = False,
                     high_confidence = True):
    
    #COnfigure Property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
         
    # Make prediction
    log_estimate = regr.predict(property_stats)
    
    # cal range
    if high_confidence:
        # wider range
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
        
    else:
        # narrow range
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
         
    return log_estimate, upper_bound, lower_bound, interval

get_log_estimate(3,20)  

# Calculation Factor to find today's dollar value
ZILLION_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLION_MEDIAN_PRICE/np.median(boston_dataset.target)

# function to get dollar values
def get_dollar_estimate(rm, ptratio, chas = False, large_range = True):
    
    if rm < 1 or ptratio < 1:
        print('This is unrealistic. Try again')
        return
    
    
    log_est, upper, lower, conf = get_log_estimate(rm, students_per_classroom = ptratio,
                                                   next_to_river = chas,
                                                   high_confidence = large_range)
    # Convert to today's dollar
    dollar_est = np.e**log_est * 1000 * SCALE_FACTOR
    dollar_hi = np.e**upper * 1000 * SCALE_FACTOR
    dollar_low = np.e**lower * 1000 * SCALE_FACTOR
    
    # Rounded dollar values to near thousand
    rounded_est = np.around(dollar_est, -3)
    rounded_hi = np.around(dollar_hi, -3)
    rounded_low = np.around(dollar_low, -3)
    
    print('The estimated property value is' ,rounded_est[0][0])
    print('At ', conf, ' confidence the valuatiion range is')
    print('USD ',rounded_low[0][0],' at the lower end to USD ',rounded_hi[0][0],' at the high end.')
    
    
get_dollar_estimate(rm =2, ptratio =30, chas = True)
 
    
    
    
    
    
    
    
























