import numpy as np

# def step_function(x):
#     return 1 if x>0 else 0

# class Perceptron():
#     def __init__(self,input_size,learning_rate=0.01) -> None:
#         self.weight = np.zeros(input_size)
#         self.bias = 0
#         self.learning_rate = learning_rate
#     def predict(self,x):
#         z = np.dot(self.weight,x)+self.bias
#         return step_function(z)
#     def train(self,x,y,epochs):
#         for epoch in range(epochs):
#             for i in range(len(x)):
#                 prediction = self.predict(x[i])
#                 error = y[i] - prediction
#                 self.weight+=self.learning_rate*error*x[i]
#                 self.bias+= self.learning_rate*error
#             print(f"Epoch {epoch+1}: Weights: {self.weight}, Bias: {self.bias}")
            
# x = np.array([[0,0],[1,0],[1,1],[0,1]])
# y = np.array([0,1,1,1])
# perceptron = Perceptron(input_size=2)
# perceptron.train(x,y,epochs=10)

# testing = np.array([[1,1]])
# for i in testing:
#     print(f"Input: {i}, Predicted Output: {perceptron.predict(i)}")

X_spam = np.random.multivariate_normal([7, 200], [[1.5, 20], [20, 3000]], size=50)
X_not_spam = np.random.multivariate_normal([3, 50], [[1.5, 20], [20, 3000]], size=50)

y_spam = np.ones(50)
y_not_spam = np.zeros(50)

X = np.vstack((X_spam, X_not_spam))
y = np.hstack((y_spam, y_not_spam))

class Peceptron():
    def __init__(self,input_size,learning_rate = 0.01) -> None:
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
    def sigmoid(self,x):
        z = np.dot(self.weights,x)+self.bias
        return 1 if z > 0 else 0
    def train(self,x,y,epochs):
        for epoch in epochs:
            for i in range(len(x)):
                prediction = self.sigmoid(x[i])
                error  = y[i] - prediction
                self.weights = self.learning_rate * error * x[i]
                self.bias = self.learning_rate*error
            print(f"for {epoch} weigth is {self.weights} and bias is {self.bias}")

print(y)