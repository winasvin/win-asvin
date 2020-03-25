
# coding: utf-8

# In[5]:


from __future__ import print_function
import sys

if __name__ == '__main__':

    infile = sys. argv [1]
    outfile = sys. argv [2]
    import numpy as np
    f= open(infile,"r")
    f2 = open(outfile, "w") 

    a = np.genfromtxt(infile,delimiter=",",dtype=str)

    b = a[1:,-1:]

    e = {}
    for c in b:
        if str(c) in e:
            e[str(c)] += 1
        else:
            e[str(c)] = 1

    higest_freq = 0
    entropy = 0
    for key in e:

        if e[key] >higest_freq:
            higest_freq = e[key]

        if ((e[key]+0.00)/(len(b))) != 0:

            entropy += -(e[key]+0.00)/len(b) *np.log((e[key]+0.00)/len(b))/np.log(2)


    error = 1 - (higest_freq+0.00)/len(b)



    f2.writelines('')
    f2.writelines('entropy: ' + str(entropy) + '\n')
    f2.writelines('error: ' + str(error) + '\n')


# In[4]:


# infile = 'handout/small_train.csv'
# outfile = 'abc.txt'

# import numpy as np
# f= open(infile,"r")
# f2 = open(outfile, "w") 

# a = np.genfromtxt(infile,delimiter=",",dtype=str)

# b = a[1:,-1:]

# e = {}
# for c in b:
#     if str(c) in e:
#         e[str(c)] += 1
#     else:
#         e[str(c)] = 1

# higest_freq = 0
# entropy = 0
# for key in e:
    
#     if e[key] >higest_freq:
#         higest_freq = e[key]
        
#     if ((e[key]+0.00)/(len(b))) != 0:
        
#         entropy += -(e[key]+0.00)/len(b) *np.log((e[key]+0.00)/len(b))/np.log(2)
        

# error = 1 - (higest_freq+0.00)/len(b)



# f2.writelines('')
# f2.writelines('entropy: ' + str(entropy) + '\n')
# f2.writelines('error: ' + str(error) + '\n')

