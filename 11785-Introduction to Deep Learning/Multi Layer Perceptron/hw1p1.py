#!/usr/bin/env python
# coding: utf-8

# In[309]:


"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):

        # Might we need to store something before returning?
        sm = 1/(1+np.exp(-x))
        self.state = sm

        return sm

    def derivative(self):

        # Maybe something we need later in here...
        sm_d = self.state * (1-self.state)

        return sm_d


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        sm = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        self.state = sm

        return sm

    def derivative(self):
        
        return 1-self.state**2
    
    
class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.x = x
        a = np.copy(x)
        a[a<0]=0
        self.state = a
        return self.state

    def derivative(self):
        a = self.x
        a[a>=0]=1
        a[a<0]=0
        return a

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):

        
        self.logits = x.astype(float)
        self.labels = y
        self.loss = np.zeros(len(x)).astype(float)
        
        
        #start my code
        

        for m in range(len(self.logits)):
            a = np.max(self.logits[m]) 
            e = 0
            for i in self.logits[m]:
                e += np.exp(i-a)
            a += np.log(e)
            deno = np.exp(a)
            aa = (np.exp(self.logits[m]))/deno
            self.logits[m][:] = aa
            loss = 0
            for b in range(len(aa)):
                if(self.labels[m][b]==1):
                    loss -= np.log(aa[b])
            self.loss[m] = loss
        self.state = self.logits
#         print(self.state)
        return self.loss

    def derivative(self):
#         print(self.logits)

        # self.sm might be useful here...

        return self.logits-self.labels

        
        
        
        

class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        # if eval:
        #    # ???

        self.x = x

#         self.mean = 
        # self.var = # ???
        # self.norm = # ???
        # self.out = # ???

        # update running batch statistics
        # self.running_mean = # ???
        # self.running_var = # ???

        # ...

        raise NotImplemented

    def backward(self, delta):

        raise NotImplemented


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return  np.random.normal(size=(d0,d1))


def zeros_bias_init(d):
    return  np.zeros(d)


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        
        self.hiddens = hiddens
        
        
        self.W = []
        self.dW = []
        self.b = []
        self.db = []
        


            
            
        if(self.nlayers == 1):
            self.W.append(weight_init_fn(self.input_size,self.output_size))

        else:
            self.W.append(weight_init_fn(self.input_size, self.hiddens[0]))

            
            if(self.nlayers > 2):
                for i in range(self.nlayers-2):
                    self.W.append(weight_init_fn( self.hiddens[i], self.hiddens[i+1]))


            self.W.append(weight_init_fn(hiddens[self.nlayers-2],self.output_size))

            
            
        if(self.nlayers == 1):
   
            self.dW.append(np.zeros((self.input_size,self.output_size)))

        else:

            self.dW.append(np.zeros((self.input_size, self.hiddens[0])))
            
            if(self.nlayers > 2):
                for i in range(self.nlayers-2):

                    self.dW.append(np.zeros(( self.hiddens[i], self.hiddens[i+1])))


            self.dW.append(np.zeros((hiddens[self.nlayers-2],self.output_size)))
            
            
        if(self.nlayers == 1):

            self.b.append(bias_init_fn(output_size))

        else:

            self.b.append(bias_init_fn( self.hiddens[0]))

            
            if(self.nlayers > 2):
                for i in range(self.nlayers-2):

                    self.b.append(bias_init_fn( self.hiddens[i+1]))
                 
            self.b.append(bias_init_fn(self.output_size))
    


        if(self.nlayers == 1):
            self.db.append(bias_init_fn(output_size))
        else:
            self.db.append(bias_init_fn( self.hiddens[0]))
            
            if(self.nlayers > 2):
                for i in range(self.nlayers-2):
                    self.db.append(bias_init_fn( self.hiddens[i+1]))
            self.db.append(bias_init_fn(self.output_size))


            
            

        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = None

        # Feel free to add any other attributes useful to your implementation (input, output, ...)

    def forward(self, x):
        
        self.x = x
        self.f = [x]
        

        
        
        fwd = None
        
        if(self.nlayers ==1):
            fwd = np.zeros((len(self.x),self.output_size))
            count = 0
            for i in self.x:
                fwd[count] = (np.matmul(i,self.W[0])+self.b[0]) 
                count += 1
            iden = Identity()
            fwd = iden.forward(fwd)
            self.f.append(fwd)
            self.y.append(iden.derivative())
                    
        else:
            fwd = np.zeros((len(self.x),self.hiddens[0]))
            count = 0
            for i in x: #next loop
                fwd[count] = (np.matmul(i,self.W[0])+self.b[0]) 
                count += 1 
            atv = self.activations[0]
            fwd = atv.forward(fwd)
            self.f.append(fwd)
            self.y.append(atv.derivative())
                
                
            if(self.nlayers >2): #middle loop
                loop = 0
                for i in range(self.nlayers-2):
                    fwd_next = np.zeros((len(self.x),self.hiddens[i+1]))
                    count = 0
                    for j in fwd:
#                         print(fwd_next.shape)
#                         print((np.matmul(self.W[loop+1] ,i)+self.b[loop+1]).shape)
                        fwd_next[count] = (np.matmul(j,self.W[loop+1])+self.b[loop+1]) 
                        count += 1 
                    atv = self.activations[i+1]
                    fwd = atv.forward(fwd_next)
                    self.f.append(fwd)
                    self.y.append(atv.derivative())
                    loop += 1
            
#             print(fwd)
                        
                        
            fwd_next = np.zeros((len(self.x),self.output_size))            
            count = 0
            for i in fwd: #final loop
                fwd_next[count] = (np.matmul(i,self.W[self.nlayers-1])+self.b[self.nlayers-1]) 
                count += 1 
            atv = self.activations[self.nlayers-1]
            fwd = atv.forward(fwd_next)
            self.f.append(fwd)
            self.y.append(atv.derivative())
                  
        self.state = fwd        
        return fwd
    
    

    def zero_grads(self):
        self.dW = np.zeros((self.output_size,self.input_size))
        self.db = np.zeros(self.output_size)

    def step(self):
        raise NotImplemented

    def backward(self, labels):
#         self.zero_grads()
        

        
        if(self.nlayers == 1):
            smce = self.criterion
            smce.forward(self.f[-1],labels)
            self.dy.append(smce.derivative())
            self.dz.append(smce.derivative())   #dz

            self.dy.append(np.matmul(self.W[-1],self.dz[-1].T).T)

            self.dW[-1] = np.matmul(self.f[0].T,self.dz[-1])/len(labels)
            self.db[-1] = np.matmul(np.ones(len(labels)),self.dz[-1])/len(labels)

        else:
            smce = self.criterion
            smce.forward(self.f[-1],labels)
            self.dy.append(smce.derivative())
            print(smce.derivative().shape)
            self.dz.append(self.y[-1]*smce.derivative())
            


            for k in reversed(range(self.nlayers-1)):

                self.dy.append(np.matmul(self.W[k+1],self.dz[-1].T).T)
                

                self.dW[k+1] = np.matmul(self.f[k+1].T,self.dz[-1])/len(labels)
                self.db[k+1] = np.matmul(np.ones(len(labels)),self.dz[-1])/len(labels)
                
                
                self.dz.append(self.dy[-1]*self.y[k])
                
            
            self.dW[0] = np.matmul(self.f[0].T,self.dz[-1])/len(labels)
            self.db[0] = np.matmul(np.ones(len(labels)),self.dz[-1])/len(labels)
                
                
                
                
        
        return None

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):

            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):

            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    for b in range(0, len(testx), batch_size):

        pass  # Remove this line when you start implementing this
        # Test ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented


# In[310]:


# t1 = np.array([[1,1,1],[2,1,3]])
# t2 = np.array([[0,1,1],[1,1,1],[1,0,0],[0,1,0]])
# b = np.array([1,2,3,4])


# In[311]:


# for i in t1:
#     print(np.matmul(t2,i)+b)


# In[312]:


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        
        self.hiddens = hiddens
        
        
        self.W = []
        self.dW = []
        self.b = []
        self.db = []
         
        self.dy = []
        self.dz = []
        self.y = []
        self.f = []
        


            
            
        if(self.nlayers == 1):
            self.W.append(weight_init_fn(self.input_size,self.output_size))

        else:
            self.W.append(weight_init_fn(self.input_size, self.hiddens[0]))

            
            if(self.nlayers > 2):
                for i in range(self.nlayers-2):
                    self.W.append(weight_init_fn( self.hiddens[i], self.hiddens[i+1]))


            self.W.append(weight_init_fn(hiddens[self.nlayers-2],self.output_size))

            
            
        if(self.nlayers == 1):
   
            self.dW.append(np.zeros((self.input_size,self.output_size)))

        else:

            self.dW.append(np.zeros((self.input_size, self.hiddens[0])))
            
            if(self.nlayers > 2):
                for i in range(self.nlayers-2):

                    self.dW.append(np.zeros(( self.hiddens[i], self.hiddens[i+1])))


            self.dW.append(np.zeros((hiddens[self.nlayers-2],self.output_size)))
            
            
        if(self.nlayers == 1):

            self.b.append(bias_init_fn(output_size))

        else:

            self.b.append(bias_init_fn( self.hiddens[0]))

            
            if(self.nlayers > 2):
                for i in range(self.nlayers-2):

                    self.b.append(bias_init_fn( self.hiddens[i+1]))
                 
            self.b.append(bias_init_fn(self.output_size))
    


        if(self.nlayers == 1):
            self.db.append(np.zeros(output_size))
        else:
            self.db.append(np.zeros( self.hiddens[0]))
            
            if(self.nlayers > 2):
                for i in range(self.nlayers-2):
                    self.db.append(np.zeros( self.hiddens[i+1]))
            self.db.append(np.zeros(self.output_size))
            
            

        self.deltaW = []
        self.deltab = []
        
        for i in range(len(self.W)):
            self.deltaW.append(np.zeros((self.W[i].shape[0],self.W[i].shape[1])))
            self.deltab.append(np.zeros(self.b[i].shape[0]))
            
#         print(self.deltaW)
            
#         print(self.deltaW[0])


            
            

        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = None

        # Feel free to add any other attributes useful to your implementation (input, output, ...)

    def forward(self, x):
        
        self.x = x
        self.f = [x]
        self.y = []
        

        
        
        fwd = None
        
        if(self.nlayers ==1):
            fwd = np.zeros((len(self.x),self.output_size))
            count = 0
            for i in self.x:
                fwd[count] = (np.matmul(i,self.W[0])+self.b[0]) 
                count += 1
            iden = Identity()
            fwd = iden.forward(fwd)
            self.f.append(fwd)
            self.y.append(iden.derivative())
                    
        else:
            fwd = np.zeros((len(self.x),self.hiddens[0]))
            count = 0
            for i in x: #next loop
                fwd[count] = (np.matmul(i,self.W[0])+self.b[0]) 
                count += 1 
            atv = self.activations[0]
            fwd = atv.forward(fwd)
            self.f.append(fwd)
            self.y.append(atv.derivative())
                
                
            if(self.nlayers >2): #middle loop
                loop = 0
                for i in range(self.nlayers-2):
                    fwd_next = np.zeros((len(self.x),self.hiddens[i+1]))
                    count = 0
                    for j in fwd:
#                         print(fwd_next.shape)
#                         print((np.matmul(self.W[loop+1] ,i)+self.b[loop+1]).shape)
                        fwd_next[count] = (np.matmul(j,self.W[loop+1])+self.b[loop+1]) 
                        count += 1 
                    atv = self.activations[i+1]
                    fwd = atv.forward(fwd_next)
                    self.f.append(fwd)
                    self.y.append(atv.derivative())
                    loop += 1
            
#             print(fwd)
                        
                        
            fwd_next = np.zeros((len(self.x),self.output_size))            
            count = 0
            for i in fwd: #final loop
                fwd_next[count] = (np.matmul(i,self.W[self.nlayers-1])+self.b[self.nlayers-1]) 
                count += 1 
            atv = self.activations[self.nlayers-1]
            fwd = atv.forward(fwd_next)
            self.f.append(fwd)
            self.y.append(atv.derivative())
                  
        self.state = fwd    

        return fwd
    
    

    def zero_grads(self):
        for i in range(len(self.dW)):
            self.dW[i] = self.dW[i]*0
            self.db[i] = self.db[i]*0
        return None


    def step(self):
        
#         print(self.deltaW[0])
        
        if(self.momentum == 0):
            for i in range(len(self.W)):
                self.W[i] = self.W[i] - self.lr*self.dW[i]
                self.b[i] = self.b[i] - self.lr*self.db[i]
        else:
            
            tW = []
            tb = []
            dtW = None
            dtb = None
            
            for i in range(len(self.W)):
#                 print('stat')
#                 print(self.deltaW[i])
#                 print(self.momentum)
#                 print(self.lr)
#                 print(self.dW[i])
                
                dtW = self.momentum*(self.deltaW[i]) - self.lr*self.dW[i] 
#                 print((self.momentum*(self.deltaW[i])).shape)
#                 print((self.lr*self.dW[i]).shape )
                tW.append(self.W[i] + dtW)
                self.deltaW[i] = dtW
    
    
#                 print('stat2')
#                 print(dtW)
#                 print(self.momentum)
#                 print(self.lr)
#                 print(self.dW[i] )
                
                dtb = self.momentum*(self.deltab[i]) - self.lr*self.db[i] 
                tb.append(self.b[i] + dtb)
                self.deltab[i] = dtb
                
            self.W = tW
            self.b = tb

        return None
    
    

    def backward(self, labels):
#         self.zero_grads()

        self.dy = []
        self.dz = []
        

        
        if(self.nlayers == 1):
            smce = self.criterion
            smce.forward(self.f[-1],labels)
            self.dy.append(smce.derivative())
            self.dz.append(smce.derivative())   #dz

            self.dy.append(np.matmul(self.W[-1],self.dz[-1].T).T)

            self.dW[-1] = np.matmul(self.f[0].T,self.dz[-1])/len(labels)
            self.db[-1] = np.matmul(np.ones(len(labels)),self.dz[-1])/len(labels)

        else:
            smce = self.criterion
            smce.forward(self.f[-1],labels)
            self.dy.append(smce.derivative())
            self.dz.append(self.y[-1]*smce.derivative())
            


            for k in reversed(range(self.nlayers-1)):

                self.dy.append(np.matmul(self.W[k+1],self.dz[-1].T).T)
                

                self.dW[k+1] = np.matmul(self.f[k+1].T,self.dz[-1])/len(labels)
                self.db[k+1] = np.matmul(np.ones(len(labels)),self.dz[-1])/len(labels)
                
                
                self.dz.append(self.dy[-1]*self.y[k])
                
            
            self.dW[0] = np.matmul(self.f[0].T,self.dz[-1])/len(labels)
            self.db[0] = np.matmul(np.ones(len(labels)),self.dz[-1])/len(labels)

                
        
        return None
    
    
    

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


# In[313]:


# mlp = MLP(3, 10,[5,7,9], [Sigmoid(),Sigmoid(),Sigmoid(),Sigmoid()],random_normal_weight_init
#           , zeros_bias_init, SoftmaxCrossEntropy(), 0.008, momentum=0.9, num_bn_layers=0)
# mlp = MLP(784, 10, [5], [Sigmoid(),Sigmoid()],
#                       random_normal_weight_init, zeros_bias_init, SoftmaxCrossEntropy(), 0.008,
#                       momentum=0.0, num_bn_layers=0)


# In[314]:


# mlp = MLP(784, 10,[64,64,32], [Sigmoid(),Sigmoid(),Sigmoid(),Identity()],random_normal_weight_init
#           , zeros_bias_init, SoftmaxCrossEntropy(), 0.008, momentum=0.9, num_bn_layers=0)


mlp = MLP(784, 10, [64, 32], [Sigmoid(),Sigmoid(),Identity()], random_normal_weight_init, zeros_bias_init, SoftmaxCrossEntropy(), 0.008,
                      momentum=0.856, num_bn_layers=0)


# In[315]:


np.random.seed(0)
t1 = np.random.randn(20,784)
t2 = np.zeros((20,10))
# mlp.forward(t1)
# mlp.backward(t2)


# In[316]:


for u in range(2):
    mlp.zero_grads()
#     print(mlp.deltaW[0])
    mlp.forward(t1)
#     print(mlp.deltaW[0])
    mlp.backward(t2)
#     print(mlp.deltaW[0])
    mlp.step()
#     print(mlp.deltaW[0])


# In[317]:



mlp.dW


# In[318]:


np.random.seed(0)
t1 = np.random.randn(20,5)
t2 = np.zeros((20,3))
mlp = MLP(5, 3, [4], [Sigmoid(),Identity()], random_normal_weight_init, zeros_bias_init, SoftmaxCrossEntropy(), 0.008,
                      momentum=0.856, num_bn_layers=0)


# In[319]:


for u in range(5):
    mlp.zero_grads()
#     print(mlp.deltaW[0])
    mlp.forward(t1)
#     print(mlp.deltaW[0])
    mlp.backward(t2)
#     print(mlp.deltaW[0])
    mlp.step()
#     print(mlp.deltaW[0])


# In[320]:


mlp.W


# In[ ]:




