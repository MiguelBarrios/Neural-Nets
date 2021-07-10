
import numpy as np

def sigmoid(x):
	return round((1/(1 + np.exp(-x))),4) 


def backprop(Z, t):
	W = np.array([[1,0,-0.8,0]])
	V = np.array([[-0.3,0.3,-0.9],[0.2,0.3,-0.9],[0.5,0.9,0.4]])
	sigmoid_v = np.vectorize(sigmoid)
	net_y =  Z[0] * np.transpose(V[0]) + Z[1] * np.transpose(V[1]) + Z[2] * np.transpose(V[2])
	y_activation = sigmoid_v(net_y)
	y_activation = np.append(y_activation,-1)
	net_out = round(np.dot(y_activation, np.transpose(W))[0],4)
	o = sigmoid(net_out)
	print("o = {}".format(o))
	print("## backprop ##\nW_old:", end = "")
	print(W)
	print("y_activation: ", end = "")
	print(y_activation)
	W = W + (t - o) * (1 - o) * o  * y_activation 
	W = np.round(W,4)
	print("W_new = ",end = "")
	print(W)

	print("V_old: ")
	print(V)
	for i in range(3):
		cur =  (t - o) * (1 - o) * o * W * (1 - y_activation) * y_activation * -1
		cur = V[i] + np.delete(cur,-1)
		V[i] = np.round(cur,4)
	print("V_new:")
	print(V)
	# calc new values
	"Re cals"
	net_y =  Z[0] * np.transpose(V[0]) + Z[1] * np.transpose(V[1]) + Z[2] * np.transpose(V[2])
	print("y_net")
	print(net_y)
	y_activation = sigmoid_v(net_y)
	print("y_activation")
	y_activation = np.append(y_activation,-1)
	print(y_activation)
	net_out = round(np.dot(y_activation, np.transpose(W))[0],4)
	print("net_out")
	print(net_out)
	print("output")
	o = sigmoid(net_out)
	print(o)



Z = [-1,-1,-1]

backprop(Z, 0.9)


