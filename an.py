import numpy as np

def fan(net):
	if net >= 0:
		return 1
	else:
		return 0
def fan2(net):
	alpha = 25
	return net * alpha

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forwardProp(Z):
	v11 = -0.3
	v21 = 0.3
	v31 = -0.9
	v12 = 0.2
	v22 = 0.3
	v32 = -0.9
	v13 = 0.5
	v23 = 0.9
	v33 = 0.4
	w1 = 1.0
	w2 = -0.4 #0
	w3 = -0.8
	w4 = 0
	V = np.array([[v11,v12,v13],[v21,v22,v23],[v31,v32,v33]])
	y1_net = round(np.dot(V[0],Z),5)
	y2_net = round(np.dot(V[1],Z),5)
	y3_net = round(np.dot(V[2],Z),5)
	fa_y1 = round(fan2(y1_net),4)
	fa_y2 = round(fan2(y2_net),4)
	fa_y3 = round(fan2(y3_net),4)
	net_out = (w1) * fa_y1 + (w2) * (fa_y2) + (w3) * fa_y3 + (w4) * -1
	net_out = round(net_out,4)
	res = fan(net_out)
	#print("y1_net = {} fan(net) = {}".format(y1_net,fa_y1), end = " : ")
	#print("y2_net = {} fan(net) = {}".format(y2_net,fa_y2), end = " : ")
	#print("y3_net = {} fan(net) = {}".format(y3_net,fa_y3))
	print("[{},{}] net_out = {} fan(net) = ##  {}  ##".format(Z[0],Z[1],net_out,res))


def backprop(z1,z2,z3):
	V = np.array([-0.3,0.3,-0.9,0.2,0.3,-0.9,0.5,0.9,0.4])
	W = np.array([1,0,-0.8,1,0,-0.8,1,0,-0.8])
	y1 = 0.4013
	y2 = 0.1824
	y3 = 0.8022
	Y = np.array([y1,y2,y3,y1,y2,y3,y1,y2,y3])
	t = 0.9
	o = 0.4402
	res = V + (t - o) * (1 - o) * o * W * (1 - Y) * Y * (-1)
	#print(V)
	print(res)


data = np.array([[-5.0,-2.0,-1],
	[5.0,-3.0,-1],
	[0.0,0.0,-1],
	[-2.0,-4.0,-1],
	[2.0,2.0,-1],
	[1.0,5.0,-1],
	[-1.0,3.0,-1]])

for i in data:
	forwardProp(i)

