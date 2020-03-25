
# coding: utf-8

# In[555]:


from __future__ import print_function
import sys
import numpy as np
import random


from environment import MountainCar

def main(args):
    mode = args[1]
    weight_out = args[2]
#     print(mode, weight_out)
    returns_out = args[3]
    episodes = int(args[4])
    max_iterations = int(args[5])
    epsilon = float(args[6])
    gamma = float(args[7])
    learning_rate = float(args[8])
    car = MountainCar(mode)
#     return car
    

#     mode = sys. argv[0]
#     weight_out = sys. argv[1]
#     returns_out = sys. argv[2]
#     episodes = int(sys. argv[3])
#     max_iterations = int(sys. argv[4])
#     epsilon = float(sys. argv[5])
#     gamma = float(sys. argv[6])
#     learning_rate = float(sys. argv[7])

#     mode = 'tile'
#     max_iterations = 200
#     episodes = 4
#     epsilon = 0.05
#     gamma = 0.99
#     learning_rate = 0.01
#     returns_out = 'r.txt'
#     weight_out = 'w.txt'


#     car = main(sys.argv)



    returns_out = open(returns_out,"w") 
    weight_out = open(weight_out,"w") 

    return_out_raw = ''
    weight_out_raw = ''


#     if mode == 'raw..':


#     #     a = car.step(0)
#         bias = 0
#         w = np.zeros((2,3)) 


#         def calc_q(state,action):
#             qsaw = state[0]*w[0][action] + state[1]*w[1][action] + bias
#     #         print('(-----)')
#     #         print(state[0])
#     #         print(w[0][action])
#     #         print(state[1])
#     #         print(w[1][action])
#     #         print(bias)
#     #         print('(-----)')

#             return qsaw

#         for i in range(episodes):
#             reward = 0
#             car.reset()

#             e = random.random()
#             if e <= epsilon:
#                 c = np.random.randint(0,3)
#             else:
#                 c = 0
#     #             c = np.argmax(np.array([calc_q(a[0],j) for j in range(3)]))
#             a0 = car.state
#             a = car.step(c)
#             d = np.array([calc_q(a[0],j) for j in range(3)])
#     #         print(d)
#     #         print(w[:,c])
#     #         print(learning_rate*(calc_q(a[0],c)-(a[1]+gamma*np.max(d))))
#     #         print([a[0][0],a[0][1]])
#     #         print(np.multiply(learning_rate*(calc_q(a[0],c)-(a[1]+gamma*np.max(d))),[a[0][0],a[0][1]]))
#     #         print('st')
#             qsa = calc_q(a0,c)
#             w[:,c] = w[:,c]- learning_rate*np.multiply((qsa-(a[1]+gamma*np.max(d))),[a0[0],a0[1]])
#             bias = bias - learning_rate*(qsa-(a[1]+gamma*np.max(d)))
#     #         print(a0)
#     #         print(c)
#     #         print(calc_q(a0,c))
#     #         print(a[1])
#     #         print(gamma*np.max(d))
#     #         print((calc_q(a0,c)-(a[1]+gamma*np.max(d))))
#     #         print('b ' + str(bias))

#     #         print(w[:,c])
#             reward += a[1]

#             while a[2] == False and abs(reward)<max_iterations:
#                 e = random.random()
#                 if e <= epsilon:
#                     c = np.random.randint(0,3)
#                 else:
#                     c = np.argmax(np.array([calc_q(a[0],j) for j in range(3)]))
#     #             print(c)
#                 a0 = a
#                 a = car.step(c)
#                 d = np.array([calc_q(a[0],j) for j in range(3)])
#                 qsa = calc_q(a0[0],c)
#                 w[:,c] = w[:,c]- learning_rate*np.multiply(qsa-(a[1]+gamma*np.max(d)),[a0[0][0],a0[0][1]])
#                 bias = bias - learning_rate*(qsa-(a[1]+gamma*np.max(d)))
#     #             print('b ' + str(bias))

#                 reward += a[1]
#             return_out_raw += str(reward) + '\n'


#         weight_out_raw += str(bias) + '\n'
#         for i in w:
#             for j in i:
#                 weight_out_raw += str(j) + '\n'

#     else:
# #     mode == 'tile':
    
    if mode == 'tile':
        s = 2048
    else:
        s = 2
    bias = 0
    w = np.zeros((s,3)) 



    def calc_q(state,action):
        qsaw = bias
        for i in state:
            qsaw += state[i]*w[i][action]
        return qsaw

    for i in range(episodes):
        reward = 0
        car.reset()

        a0 = car.transform(car.state)
        e = random.random()
        if e <= epsilon:
            c = np.random.randint(0,3)
        else:
            c = np.argmax(np.array([calc_q(a0,j) for j in range(3)]))


        a = car.step(c)
        d = np.array([calc_q(a[0],j) for j in range(3)])
        qsa = calc_q(a0,c)
        kk = np.zeros((1,s))
        for k in a0:
#                 kk[0][k] = 1
            kk[0][k] = a0[k]
#         print(kk)
        w[:,c] = w[:,c]- learning_rate*np.multiply(qsa-(a[1]+gamma*np.max(d)),kk)
        bias = bias - learning_rate*(qsa-(a[1]+gamma*np.max(d)))
#         print(bias)
#         print(qsa)
#         print(a[1]+gamma*np.max(d))
#         print(bias)

        reward += a[1]

        while a[2] == False and abs(reward)<max_iterations:
            e = random.random()
            if e <= epsilon:
                c = np.random.randint(0,3)
            else:
                c = np.argmax(np.array([calc_q(a[0],j) for j in range(3)]))
#             print(c)
            a0 = a
            a = car.step(c)
            d = np.array([calc_q(a[0],j) for j in range(3)])
            qsa = calc_q(a0[0],c)
            kk = np.zeros((1,s))
            for k in a0[0]:
                kk[0][k] = a0[0][k]
#                     kk[0][k] = 1
            w[:,c] = w[:,c]- learning_rate*np.multiply(qsa-(a[1]+gamma*np.max(d)),kk)
            bias = bias - learning_rate*(qsa-(a[1]+gamma*np.max(d)))


#             print('b ' + str(bias))

            reward += a[1]
        return_out_raw += str(reward) + '\n'


    weight_out_raw += str(bias) + '\n'
    for i in w:
        for j in i:
            weight_out_raw += str(j) + '\n'



#     print(return_out_raw)
#     print(weight_out_raw)

    
    returns_out.writelines(return_out_raw)
    weight_out.writelines(weight_out_raw)
    
    

if __name__ == "__main__":
#     print('test')
#     sys.args = ['a','b']
#     sys.argv = ['q_learning.py','tile', 'q42_weight.out', 'q42_returns.out', '4', '200', '0.05', '0.99', '0.01']
#     sys.argv = ['q_learning.py','raw', 'q42_weight.out', 'q42_returns.out', '100', '200', '0.0', '0.9', '0.01']
    main(sys.argv)
       
    
    

