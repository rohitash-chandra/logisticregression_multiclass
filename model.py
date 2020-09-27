 
 # Updated by R. Chandra

from math import exp
import numpy as np

SIGMOID = 1
STEP = 2
LINEAR = 3

class logistic_regression:

	def __init__(self, num_epocs, train_data, num_features, learn_rate):
		self.train_data = train_data
		self.num_features = num_features
		self.num_outputs = self.train_data.shape[1] - num_features
		self.num_train = self.train_data.shape[0]
		self.w = np.random.uniform(0, 0, num_features)  
		self.b = np.random.uniform(0, 0, self.num_outputs) 
		self.learn_rate = learn_rate
		self.max_epoch = num_epocs
		self.use_activation = SIGMOID # 1 is  sigmoid , 2 is step, 3 is linear
		self.out_delta = np.zeros(self.num_outputs)

		print(self.w, ' self.w') 
		print(self.b, ' self.b') 
		print(self.out_delta, ' outdel')
 
	def activation_func(self,z_vec):
		if self.use_activation == SIGMOID:
			y =  1 / (1 + np.exp(z_vec)) # sigmoid/logistic
		elif self.use_activation == STEP: 
			if z_vec > 0: #step function (need to generalise it to multiclass)
				y = 1
			else: y = 0
		else:
			z = z_vec
		return y
 

	def predict(self, x_vec ): 
		z_vec = x_vec.dot(self.w) - self.b 
		output = self.activation_func(z_vec) # Output  
		return output
	
	def gradient(self, x_vec, output, actual):   
		if self.use_activation == SIGMOID :
			out_delta =   (output - actual)*(output*(1-output)) 
		else: # for linear and step function  
			out_delta =   (output - actual) 
		return out_delta

	def update(self, x_vec, output, actual):      
		self.w+= self.learn_rate *( x_vec *  self.out_delta)
		self.b+=  (1 * self.learn_rate * self.out_delta)
 

	def squared_error(self, prediction, actual):
		return  np.sum(np.square(prediction - actual))/prediction.shape[0]# to cater more in one output/class

	def test_model(self):  


	 
 
	def SGD(self):   
		
			epoch = 0 
			while  epoch < self.max_epoch:
				sum_sqer = 0
				for s in range(0, self.num_train): 
					input_instance  =  self.train_data[s,0:self.num_features]  
					actual  = self.train_data[s,self.num_features:]  
					prediction = self.predict(input_instance) 
					sum_sqer += self.squared_error(prediction, actual)
					self.out_delta = self.gradient( input_instance, prediction, actual)    # major difference when compared to GD
					#print(input_instance, prediction, actual, s, sum_sqer)
					self.update(input_instance, prediction, actual)

			
				print(epoch, sum_sqer, self.w, self.b)
				epoch=epoch+1  
 

			train_perc = 0
			test_perc = 0

			return (train_perc, test_perc, sum_sqer) 

	def GD(self):   
		
			epoch = 0 
			while  epoch < self.max_epoch:
				sum_sqer = 0
				for s in range(0, self.num_train): 
					input_instance  =  self.train_data[s,0:self.num_features]  
					actual  = self.train_data[s,self.num_features:]   
					prediction = self.predict(input_instance) 
					sum_sqer += self.squared_error(prediction, actual) 
					self.out_delta+= self.gradient( input_instance, prediction, actual)    # this is major difference when compared with SGD

					#print(input_instance, prediction, actual, s, sum_sqer)
				self.update(input_instance, prediction, actual)

			
				print(epoch, sum_sqer, self.w, self.b)
				epoch=epoch+1  
 

			train_perc = 0
			test_perc = 0

			return (train_perc, test_perc, sum_sqer) 
				
	
 

#------------------------------------------------------------------


 
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]


dataset_onehot = [[2.7810836,2.550537003,0, 1],
	[1.465489372,2.362125076,0, 1],
	[3.396561688,4.400293529,0, 1],
	[1.38807019,1.850220317,0, 1],
	[3.06407232,3.005305973,0, 1],
	[7.627531214,2.759262235,1, 0],
	[5.332441248,2.088626775,1, 0],
	[6.922596716,1.77106367,1, 0],
	[8.675418651,-0.242068655,1, 0],
	[7.673756466,3.508563011,1, 0]]


learn_rate = 0.3
num_features = 2
num_epocs = 30
train_data = np.asarray(dataset) # convert list data to numpy

print(train_data)
 

lreg = logistic_regression(num_epocs, train_data, num_features, learn_rate)
(train_perc, train_perc, sse) = lreg.SGD()
 
print(train_perc, train_perc, sse)


(train_perc, train_perc, sse) = lreg.GD()
 
print(train_perc, train_perc, sse)


train_data = np.asarray(dataset_onehot) # convert list data to numpy
 
print(train_data)
 

lreg = logistic_regression(num_epocs, train_data, num_features, learn_rate)
(train_perc, train_perc, sse) = lreg.SGD()
 
print(train_perc, train_perc, sse)


(train_perc, train_perc, sse) = lreg.GD()
 
print(train_perc, train_perc, sse)
