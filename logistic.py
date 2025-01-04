import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score


data=pd.read_csv('/path/to/Heart_disease.csv')
data = data.dropna() 

scaler=StandardScaler()
x=scaler.fit_transform(data[['age','currentSmoker','cigsPerDay','heartRate','BMI','totChol']].to_numpy())
y=data['TenYearCHD'].values
m,n=x.shape
w=np.zeros(n)
b=0.0
learning_rate=0.005
iterations=10000
lamda=0


def compute_gradient(x, y, w, b):
    dj_w = np.zeros(n)
    dj_b = 0
    for i in range(m):
        c = np.dot(w, x[i]) + b
        fwb = expit(c)
        error = fwb - y[i]
        dj_w += error * x[i]
        dj_b += error
    dj_w = dj_w / m + (lamda / m) * w
    dj_b /= m
    return dj_w, dj_b




def cost_function(x,y,w,b,lamda):
    cost=0
    for i in range(m):
        c=np.dot(w,x[i]) +b
        fwb = expit(c)
        epsilon = 1e-15
        d = np.log(np.clip(fwb, epsilon, 1 - epsilon))
        f = np.log(np.clip(1 - fwb, epsilon, 1 - epsilon))
        cost+=-(y[i]*d+(1-y[i])*f)
    regularization = (lamda / (2 * m)) * np.sum(w**2)
    cost=cost/m + regularization
    return cost

def gradient_desent(x,y,w,b,learning_rate,iterations,lamda):
    cost_H=[]
    for i in range(iterations):
        wg,bg=compute_gradient(x,y,w,b)
        w-=learning_rate*wg 
        b-=learning_rate*bg
        cost=cost_function(x,y,w,b,lamda)
        cost_H.append(cost)
        if i%100==0:
            print(f"cost:{cost:.2f},w:{w},b:{b:.2f}")
    return w,b,cost_H


wo,bo,cost=gradient_desent(x,y,w,b,learning_rate,iterations,lamda)
def model(x,y,w,b):
    c=np.dot(x,w) +b
    fwb=1/(1+np.exp(-c))
    return fwb
predicted=model(x,y,wo,bo)



plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost over reduction")
plt.plot(cost)
plt.show()

x_axis=x[:,0]
plt.xlabel("Feature (e.g., Age)")
plt.ylabel("Probability of TenYearCHD")
plt.scatter(x_axis,y,label='Actual Values',color='red',alpha=0.5)
plt.scatter(x_axis,predicted,label='Predicted Values',color='green',alpha=0.5)
plt.legend()
plt.title("Logistic Regression")
plt.show()