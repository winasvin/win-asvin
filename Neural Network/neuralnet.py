
# coding: utf-8

# In[546]:


from __future__ import print_function
import sys

if __name__ == '__main__':

    train_input = sys. argv [1]
    test_input= sys. argv [2]
    train_out= sys. argv [3]
    test_out = sys. argv [4]
    metrics_out = sys. argv [5]
    num_epoch = int(sys. argv [6])
    hidden_units = int(sys. argv [7])
    init_flag = int(sys. argv [8])
    learning_rate = float(sys. argv [9])


    import numpy as np
    import math


#     train_input = 'handout/smallTrain.csv'
#     test_input = 'handout/smallTest.csv'
#     train_out = 'handout/to1.labels'
#     test_out = 'handout/to2.labels'
#     metrics_out = 'handout/to3.txt'
#     num_epoch = 2
#     hidden_units = 4
#     init_flag = 2
#     learning_rate = 0.1

    
    
    train_input = open(train_input,"r")
    test_input = open(test_input,"r")  
    train_out= open(train_out,"w") 
    test_out= open(test_out,"w") 
    metrics_out = open(metrics_out,"w") 


    x=[]
    y=[]
    x_test=[]
    y_test=[]
    train_out_raw = ''
    test_out_raw = ''
    metrics_out_raw = ''




    with train_input as f:
        for n in f:
            nn = (np.array((n.replace('\n','').split(',')),dtype = float))
            y.append(nn[0])
            nn[0] = 1
            x.append(nn)

    with test_input as f2:
        for n in f2:
            nn = (np.array((n.replace('\n','').split(',')),dtype = float))
            y_test.append(nn[0])
            nn[0] = 1
            x_test.append(nn)

    y_vec = []
    for i in y:
        a = np.zeros(10)
        a[int(i)] = 1
        y_vec.append(a)

    y_vec_test = []    
    for i in y_test:
        a = np.zeros(10)
        a[int(i)] = 1
        y_vec_test.append(a)


    def innerprod(x,y):
        inner = 0
        for i in range(len(x)):
            inner += x[i]*y[i]
        return inner


    if init_flag == 1:
        al = np.random.random((hidden_units,129))
        be = np.random.random((10,hidden_units+1))
        alpha = (al-0.5)/5
        beta = (be-0.5)/5
        alpha[:,0] = 0
        beta[:,0] = 0
        
#         print(alpha)
#         print(beta)
    elif init_flag == 2:
        alpha = np.zeros(129*hidden_units).reshape(hidden_units,129)        
        beta = np.zeros((hidden_units+1)*10).reshape(10,hidden_units+1)   
        




    for m in range((num_epoch)):
        for n in range(len(x)):
            a = []
            for i in range(hidden_units):
                a.append(innerprod(x[n],alpha[i]))
#             print('a',a)
        #         print(alpha[:,i])

            z = [1/(1+np.exp(-i)) for i in a]
            z = [1.] + z
#             print('z',z)

            # b = []
            # for i in range(10):
            #     b.append(innerprod(z,beta[i]))

            b = np.matmul(z,beta.T)
#             print('b',b)

            bb = 0
            for i in range(len(b)):
                bb += np.exp(b[i])

            y_hat = [np.exp(i)/bb for i in b]

#             print('y_hat',y_hat)


            gJ = 1
            gy = [-y_vec[n][i]/y_hat[i] for i in range(10)]
#             print('gy',gy)

        #     gb = [-y_vec[n][i]*(1-y_hat[j]) for i in range(10)]

            dy_db = np.ndarray((10,10))
            for i in range(10):
                for j in range(10):
                    if i == j:
                        dy_db[i][i] = (1-y_hat[i])*y_hat[i] 
                    else: 
                        dy_db[i][j] = (-y_hat[i])*y_hat[j]

            gb = np.matmul(np.array(gy),dy_db)
#             print('gb',gb)

            gbeta = np.ndarray(len(gb)*len(z)).reshape(len(z),len(gb))
            for i in range(len(z)):
                gbeta[i] = gb*z[i]
            gbeta = gbeta.T
#             print('gbeta',gbeta)

            gz = np.matmul(beta.T,gb)
#             print('gz',gz)


#             a_back = np.array(a)
#             a_back = np.exp(-a_back)/(np.exp(-a_back)+1)**2

#             dz_da = []
#             for i in range(hidden_units):
#                 dz_da.append(a_back)
#             dz_da = np.array(dz_da)
            dz_da = np.zeros((len(a),len(a)))
            for i in range(len(a)):
                for j in range(len(a)):
                    if i==j:
                        dz_da[i][j] = np.exp(-a[i])/(np.exp(-a[i])+1)**2





            
#             print('dz_da',dz_da)
            ga = np.matmul(gz[1:],dz_da.T) ##fixed
#             print('ga',ga)

            galpha = np.ndarray(hidden_units*len(x[n])).reshape(hidden_units,len(x[n]))
            for i in range(hidden_units):
                galpha[i] = ga[i]*x[n]
#             print('galpha',galpha)



            alpha = alpha - learning_rate * galpha
#             print(alpha)
            beta = beta - learning_rate * gbeta
#             print(beta)


    ##train prediction
        cross_entropy = 0
        error = 0


        for n in range(len(x)):
            a = []
            for i in range(hidden_units):
                a.append(innerprod(x[n],alpha[i]))


            z = [1/(1+np.exp(-i)) for i in a]
            z = [1.] + z


            b = np.matmul(z,beta.T)


            bb = 0
            for i in range(len(b)):
                bb += np.exp(b[i])

            y_hat = [np.exp(i)/bb for i in b]

        #     print('y_hat',y_hat)

            max_predict = 0
            predict = 0
            for i in range(10):
                if y_hat[i] >max_predict:
                    max_predict = y_hat[i]
                    predict = i

            if m == num_epoch-1:
                train_out_raw += str(predict) + '\n'

            cross_entropy -= np.log(y_hat[int(y[n])])
            if predict != int(y[n]):
                error += 1


        cross_entropy /= len(x)
        error /= len(x)


        metrics_out_raw += 'epoch='+ str(m+1) + ' crossentropy(train): ' + str(cross_entropy) + '\n'
    #     print('error',error)

    ##test prediction
        cross_entropy_test = 0
        error_test = 0


        for n in range(len(x_test)):
            a = []
    #         error = 0
            for i in range(hidden_units):
                a.append(innerprod(x_test[n],alpha[i]))


            z = [1/(1+np.exp(-i)) for i in a]
            z = [1.] + z


            b = np.matmul(z,beta.T)


            bb = 0
            for i in range(len(b)):
                bb += np.exp(b[i])

            y_hat = [np.exp(i)/bb for i in b]

        #     print('y_hat',y_hat)

            max_predict = 0
            predict = 0
            for i in range(10):
                if y_hat[i] >max_predict:
                    max_predict = y_hat[i]
                    predict = i

            if m == num_epoch-1:
                test_out_raw += str(predict) + '\n'

            cross_entropy_test -= np.log(y_hat[int(y_test[n])])
            if predict != int(y_test[n]):
                error_test += 1


        cross_entropy_test /= len(x_test)
        error_test /= len(x_test)


        metrics_out_raw += 'epoch='+ str(m+1) + ' crossentropy(test): ' + str(cross_entropy_test) + '\n'
    #     print('error',error)

    metrics_out_raw += 'error(train): ' + str(error) +'\n'
    metrics_out_raw += 'error(test): ' + str(error_test) +'\n'



    train_out.writelines(train_out_raw)
    test_out.writelines(test_out_raw)
    metrics_out.writelines(metrics_out_raw)


# In[545]:


# print(metrics_out_raw)


# In[542]:


# print(test_out_raw)

