
# coding: utf-8

# In[145]:


import numpy as np

HIDDEN_DIM = 4

class Sigmoid:
    # DO NOT DELETE
    def __init__(self):
        pass
    def forward(self, x):
        self.res = 1/(1+np.exp(-x))
        return self.res
    def backward(self):
        return self.res * (1-self.res)
    def __call__(self, x):
        return self.forward(x)


class Tanh:
# DO NOT DELETE
    def __init__(self):
        pass
    def forward(self, x):
        self.res = np.tanh(x)
        return self.res
    def backward(self):
        return 1 - (self.res**2)
    def __call__(self, x):
        return self.forward(x)

class Linear():
    # DO NOT DELETE
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.W = np.random.randn(out_feature, in_feature)
        self.b = np.zeros(out_feature)
        
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.out = x.dot(self.W.T) + self.b
        return self.out

    def backward(self, delta):
        self.db = delta
        self.dW = np.dot(self.x.T, delta)
        dx = np.dot(delta, self.W.T)
        return dx

class GRU_Cell:
    """docstring for GRU_Cell"""
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t=0

        self.Wzh = np.random.randn(h,h)
        self.Wrh = np.random.randn(h,h)
        self.Wh  = np.random.randn(h,h)

        self.Wzx = np.random.randn(h,d)
        self.Wrx = np.random.randn(h,d)
        self.Wx  = np.random.randn(h,d)

        self.dWzh = np.zeros((h,h))
        self.dWrh = np.zeros((h,h))
        self.dWh  = np.zeros((h,h))

        self.dWzx = np.zeros((h,d))
        self.dWrx = np.zeros((h,d))
        self.dWx  = np.zeros((h,d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here
        self.z = None
        self.r = None
        self.h_t1 = None
        self.h_t = None
        self.hh = None
        self.x = None
        
        


    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx  = Wx

    def __call__(self, x, h):
        return self.forward(x,h)

    def forward(self, x, h):
        # input:
        # 	- x: shape(input dim),  observation at current time-step
        # 	- h: shape(hidden dim), hidden-state at previous time-step
        # 
        # output:
        # 	- h_t: hidden state at current time-step
        
#         print('d',self.d)
#         print('h',self.h)
#         print('x',x.shape)
#         print('h',h.shape)
        
#         sigmoid = Sigmoid()
#         tanh = Tanh()
        self.x = x
        self.h_t1 = h
    
        z = self.z_act.forward(self.Wzx.dot(x) + self.Wzh.dot(h))
        r = self.r_act.forward(self.Wrx.dot(x) + self.Wrh.dot(h))
        hh = self.h_act.forward(self.Wx.dot(x) + self.Wh.dot(h * r))
        h = (np.ones(z.shape[0]) - z) * h + z * hh
        
        
        self.z = z
        self.r = r
        self.h_t = h
        self.hh = hh
        
        

        return h


# This  must calculate the gradients wrt the parameters and returns the derivative wrt the inputs, xt and ht, to the cell.
    def backward(self, delta):
    # input:
    #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
    #            the same time-step and derivative wrt loss from same layer at
    #            next time-step
    # output:
    #  - dx: Derivative of loss wrt the input x
    #  - dh: Derivative  of loss wrt the input hidden h
    
       
 

        b2 = (1-self.hh*self.hh)*(self.z * delta)
        b1 = (self.z * (1-self.z))*(self.h_t1 * delta-1*(self.hh * delta))
        c1 = ((b2.dot(self.Wh)) * self.h_t1) * (self.r * (np.ones(self.r.shape[0])-self.r))
        dx = c1.dot(self.Wrx)+(b2.dot(self.Wx)) + (-1*(b1.dot(self.Wzx)))
        dh_t1 = ((b1.dot(self.Wzh))*-1) + (b2.dot(self.Wh)) * self.r + (c1.dot(self.Wrh))+(1-self.z) * delta 
        

        
        self.dWzx = -1*(self.x.reshape((len(self.x)),1)).dot(b1).T
        self.dWrx = (self.x.reshape((len(self.x)),1)).dot(c1).T
        self.dWx  = (self.x.reshape((len(self.x)),1)).dot(b2).T
        

        
        self.dWzh = -1*(self.h_t1.reshape((len(self.h_t1)),1)).dot(b1).T
        self.dWrh = (self.h_t1.reshape((len(self.h_t1)),1)).dot(c1).T
        a1 = self.h_t1*self.r
        self.dWh  = (a1.reshape(len(a1),1)).dot(b2).T
        
        
                
#         print('dW')
#         print(self.dWzh)
#         print(self.dWrh)
#         print(self.dWh)
#         print(self.dWzx)
#         print(self.dWrx)
#         print(self.dWx)



        
    
        return dx,dh_t1

# This is the neural net that will run one timestep of the input 
# You only need to implement the forward method of this class. 
# This is to test that your GRU Cell implementation is correct when used as a GRU.	
class CharacterPredictor(object):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()
        # The network consists of a GRU Cell and a linear layer  
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.rnn = GRU_Cell(self.input_dim,self.hidden_dim)
        self.linear = Linear(self.hidden_dim,self.num_classes)
        
        

    def init_rnn_weights(self, w_hi, w_hr, w_hn, w_ii, w_ir, w_in):
        # DO NOT MODIFY
        self.rnn.init_weights(w_hi, w_hr, w_hn, w_ii, w_ir, w_in) 

    def __call__(self, x, h):
        return self.forward(x, h)        

    def forward(self, x, h):
        # A pass through one time step of the input 
        
        
        h_t = self.rnn.forward(x,h)
        logit = self.linear.forward(h_t)
        
        return h_t,logit

# An instance of the class defined above runs through a sequence of inputs to generate the logits for all the timesteps. 
def inference(net, inputs):
    # input:
    #  - net: An instance of CharacterPredictor
    #  - inputs - a sequence of inputs of dimensions [seq_len x feature_dim]
    # output:
    #  - logits - one per time step of input. Dimensions [seq_len x num_classes]
    
#     print(inputs[0])
    
    
    output = []
    h = np.zeros(net.rnn.h)
    for i in range(len(inputs)):  

        h,logit = net.forward(inputs[i],h)
        output.append(logit)
    
    output = np.array(output)


    
    return output





# In[146]:


# np.random.seed(0)
# input_dim = 5
# hidden_dim = 2
# seq_len = 10
# data = np.random.randn(seq_len, input_dim)
# hidden = np.random.randn(hidden_dim)

# g1 = GRU_Cell(input_dim, hidden_dim)


# o1 = g1.forward(data[0], hidden)
# np.random.seed(0)
# delta = np.random.randn(hidden_dim)
# delta
# delta = delta.reshape(1, -1)
# g1.backward(delta)


# In[147]:


FEATURE_DIM = 7
HIDDEN_DIM = 4
NUM_CLASSES = 3

np.random.seed(11785)

done = False
# inputs = create_input_data()#np.load("input_data.npy")

w_ir = np.random.randn(HIDDEN_DIM, FEATURE_DIM)#np.load("wir.npy")
w_ii = np.random.randn(HIDDEN_DIM, FEATURE_DIM)#np.load("wii.npy")
w_in = np.random.randn(HIDDEN_DIM, FEATURE_DIM)#np.load("win.npy")
w_hr = np.random.randn(HIDDEN_DIM, HIDDEN_DIM)#np.load("whr.npy")
w_hi = np.random.randn(HIDDEN_DIM, HIDDEN_DIM)#np.load("whi.npy")
w_hn = np.random.randn(HIDDEN_DIM, HIDDEN_DIM)#np.load("whn.npy")


# Load weights into student implementation
student_net = CharacterPredictor(FEATURE_DIM, HIDDEN_DIM, NUM_CLASSES)
np.random.seed(11785)
student_net.init_rnn_weights(w_hi, w_hr, w_hn, w_ii, w_ir, w_in)



# In[148]:


inference(student_net,np.random.randn(10, FEATURE_DIM))

