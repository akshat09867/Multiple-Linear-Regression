import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler




data=pd.read_csv('/path/to/Student_Performance.csv')


scaler = StandardScaler()
x = scaler.fit_transform(data[['Hours Studied', 'Previous Scores',  'Sample Question Papers Practiced', 'Sleep Hours']].to_numpy())
y=data['Performance Index'].values
learning_rate=0.01
iterations=1000
m,n=x.shape
w=np.zeros(n)
b=0.0


def compute_gradient(x,y,w,b):
    dj_b=0
    dj_w=np.zeros(n)
    for i in range(m):
        err=np.dot(w,x[i])+b-y[i]
        dj_b+=err
        dj_w+=err*x[i]
    dj_b/=m
    dj_w/=m
    return dj_w,dj_b


def cost_fun(x,y,w,b):
    cost=0
    for i in range(m):
        err=(np.dot(w,x[i])+b-y[i])**2
        cost+=err
    cost/=2*m
    return cost


def gradient_desent(x,y,w,b,learning_rate,iterations):
    cost_H=[]
    for i in range(iterations):
        gw,gb=compute_gradient(x,y,w,b)
        w-=learning_rate*gw
        b-=learning_rate*gb
        cost=cost_fun(x,y,w,b)
        cost_H.append(cost)
        if i%100==0:
            print(f"cost={cost:.4f}, w={[f'{weight:.3f}' for weight in w]}, b={b:.2f}")
    return w,b,cost_H




wo,bo,cost_o=gradient_desent(x,y,w,b,learning_rate,iterations)
plt.plot(cost_o)
plt.xlabel('iterations')
plt.ylabel('cost')
plt.title("cost reduction over iterations")
plt.show()


def model(x, w, b):
    return np.dot(x, w) + b  # Vectorized computation for all samples
predicted_performance = model(x,wo,bo)



# Extracting the feature for the x-axis (Hours Studied)
x_axis2=x[:,1]
# Plot the scatter for actual values
plt.scatter(x_axis2, y, label="Actual Value", color="yellow")
# Plot the predicted values
plt.plot(x_axis2, predicted_performance, label="Predicted Value", color="red")

# Labeling the plot
plt.xlabel("Hours Studied")
plt.ylabel("Performance Index")
plt.legend()
plt.title("Model Accuracy")
plt.show()
