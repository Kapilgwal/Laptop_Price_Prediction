import pickle
import numpy as np
import re
import sklearn
# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))


# brand
company = 'Apple'

# type of laptop
type = 'Ultrabook'

# Ram
ram = 8

# weight
weight = 1.83

# Touchscreen
touchscreen = 'No'

# IPS
ips = 'Yes'

# screen size
screen_size = 15.6

# resolution
X_res = 1920
Y_res = 1080
#cpu
cpu = 'Intel Core i5'

hdd = 0

ssd = 512

gpu = 'Intel'

os = 'Windows'


ppi = None
if touchscreen == 'Yes':
    touchscreen = 1
else:
    touchscreen = 0

if ips == 'Yes':
    ips = 1
else:
    ips = 0

ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

query = query.reshape(1,12)
text = str((np.exp(pipe.predict(query)[0])))
print(text)
    # st.title("The predicted price of this configuration is " + result )