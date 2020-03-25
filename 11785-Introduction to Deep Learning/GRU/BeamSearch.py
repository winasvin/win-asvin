
# coding: utf-8

# In[1]:


def GreedySearch(SymbolSets, y_probs):
	'''
	SymbolSets: This is the list containing all the symbols i.e. vocabulary (without 				  blank)
	y_probs: Numpy array of (# of symbols+1,Seq_length,batch_size). 
    Note that your 			   batch size for part 1 would always remain 1, but if you plan to use 			
    your implementation for part 2 you need to incorporate batch_size.

	Return the forward probability of greedy path and corresponding compressed symbol 	  
    sequence i.e. without blanks and repeated symbols.
	'''

def BeamSearch(SymbolSets, y_probs, BeamWidth):
	'''
	SymbolSets: This is the list containing all the symbols i.e. vocabulary (without 				  blank)
	
	y_probs: Numpy array of (# of symbols+1,Seq_length,batch_size). Note that your 			   batch size for part 1 would always remain 1, but if you plan to use 			your implementation for part 2 you need to incorporate batch_size.
	
	BeamWidth: Width of the beam.
	
	The function should return the symbol sequence with the best path score (forward 	  probability) and a dictionary of all the final merged paths with their scores. 
	'''


# In[2]:


import numpy as np
def GreedySearch(SymbolSets, y_probs):
#     print(SymbolSets)
#     print(y_probs.shape)
    prob = 1
    compressed = ''
    y_prev = ''
    for n in range(y_probs.shape[1]):
        seq = np.argmax(y_probs[:,n,0])
        prob *= y_probs[seq,n]
        print(seq)
        
        if (y_prev != seq and seq != 0):
            compressed += SymbolSets[seq-1]
        print(compressed)

        y_prev = seq


    return compressed,prob
        


# In[3]:


# s = ['a','b','c','d']
# a = np.random.rand(5,10,1)
# GreedySearch(s,a)


# In[4]:


# global BlankPathScore 
# global PathScore 
# PathScore = {}
# BlankPathScore = {}

def BeamSearch(SymbolSet, y, BeamWidth):

    global BlankPathScore 
    global PathScore 
    PathScore = {}
    BlankPathScore = {}
   

    # First time instant: Initialize paths with each of the symbols,
    # including blank, using score at time t=1
    PathsWithTerminalBlank, PathsWithTerminalSymbol, PathScore, BlankPathScore = InitializePaths(SymbolSet, y[:,0], BeamWidth)

    
    # Subsequent time steps
    for t in range(1,y.shape[1]):

    # First extend paths by a blank
#         print('path',PathScore)
        UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y[:,t])
        # Next extend paths by a symbol
#         print('path',PathScore)
  
        UpdatedPathsWithTerminalSymbol, UpdatedPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol,SymbolSet,y[:,t])
        # Prune the collection down to the BeamWidth


        PathsWithTerminalBlank, PathsWithTerminalSymbol, PathScore, BlankPathScore = Prune(UpdatedPathsWithTerminalBlank, UpdatedPathsWithTerminalSymbol,
                                             UpdatedBlankPathScore, UpdatedPathScore, BeamWidth)

        

    
    # Merge identical paths differing only by the final blank
    MergedPaths, FinalPathScore = MergeIdenticalPaths(PathsWithTerminalBlank, PathsWithTerminalSymbol)
    # Pick best path
    
    bestscore = 0
    bestpath = ''
    for k,v in FinalPathScore.items():
        if v > bestscore:
            bestscore = v
            bestpath = k
        
        
#     BestPath = np.argmax(FinalPathScore) # Find the path with the best score
    
    return k,FinalPathScore
  
    
    
    
    
    


# In[5]:


# global BlankPathScore 
# global PathScore 
def InitializePaths(SymbolSet, y, BeamWidth):
# First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
    global BlankPathScore 
    global PathScore 


    
    path = ''
    BlankPathScore[path] = y[0] # Score of blank at t=1 
    InitialPathsWithFinalBlank = set()
    InitialPathsWithFinalBlank.add(path)

    # Push rest of the symbols into a path-ending-with-symbol stack
    InitialPathsWithFinalSymbol = set()
    i =0 
    for c in SymbolSet: # This is the entire symbol set, without the blank
        path = c
        PathScore[path] = y[i+1] # Score of symbol c at t=1 
        InitialPathsWithFinalSymbol.add(path) # Set addition
        i+=1
        
#     print('InitializePaths',PathScore)
    # Prune poor paths and return
    return Prune(InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, BlankPathScore, PathScore, BeamWidth)


# In[6]:


# global BlankPathScore 
# global PathScore 
def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth): 
#     global BlankPathScore 
#     global PathScore 
    PrunedBlankPathScore = {}
    PrunedPathScore = {}
    scorelist = {}
    scorelists = []

    # First gather all the relevant scores
    i=0
    for p in PathsWithTerminalBlank:
        scorelist[i] = BlankPathScore[p]
        scorelists.append(BlankPathScore[p])
        i+=1

    for p in PathsWithTerminalSymbol:
        scorelist[i] = PathScore[p]
        scorelists.append(PathScore[p])
        i+=1
        
        

 

    # Sort and find cutoff score that retains exactly BeamWidth paths
#     sort(scorelist)  # In decreasing order
    scorelists.sort(reverse = True)

    cutoff = scorelists[BeamWidth]

    PrunedPathsWithTerminalBlank = set()

    for p in PathsWithTerminalBlank:
        if BlankPathScore[p] > cutoff:
            PrunedPathsWithTerminalBlank.add(p) # Set addition 
            PrunedBlankPathScore[p] = BlankPathScore[p]
  
    
    PrunedPathsWithTerminalSymbol = set()
    for p in PathsWithTerminalSymbol:
        if PathScore[p] > cutoff:
            PrunedPathsWithTerminalSymbol.add(p) # Set addition 
            PrunedPathScore[p] = PathScore[p]


    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedPathScore, PrunedBlankPathScore


# In[7]:


def MergeIdenticalPaths(PathsWithTerminalBlank, PathsWithTerminalSymbol):
    global BlankPathScore 
    global PathScore 
    FinalPathScore = {}
    # All paths with terminal symbols will remain
    MergedPaths = PathsWithTerminalSymbol
    for p in MergedPaths:
        FinalPathScore[p] = PathScore[p]

    # Paths with terminal blanks will contribute scores to existing identical paths from # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] += BlankPathScore[p]
        else:
            MergedPaths.add(p) # Set addition
            FinalPathScore[p] = BlankPathScore[p]
    return MergedPaths, FinalPathScore


# In[8]:


def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y):

    global BlankPathScore 
    global PathScore 

#     print('ExtendWithBlank path',PathScore,BlankPathScore)
    UpdatedPathsWithTerminalBlank = set()
    UpdatedBlankPathScore = {}
    # First work on paths with terminal blanks
    #(This represents transitions along horizontal trellis edges for blanks)
    for path in PathsWithTerminalBlank:
    # Repeating a blank doesn’t change the symbol sequence 
        UpdatedPathsWithTerminalBlank.add(path) # Set addition 
        UpdatedBlankPathScore[path] = BlankPathScore[path]*y[0]

    # Then extend paths with terminal symbols by blanks
    for path in PathsWithTerminalSymbol:
    # If there is already an equivalent string in UpdatesPathsWithTerminalBlank # simply add the score. If not create a new entry
        if path in UpdatedPathsWithTerminalBlank:
            
            UpdatedBlankPathScore[path] += PathScore[path]* y[0]
        else:
            UpdatedPathsWithTerminalBlank.add(path) # Set addition
            UpdatedBlankPathScore[path] = PathScore[path] * y[0]


    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore


# In[9]:


def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol,SymbolSet, y):

    global BlankPathScore 
    global PathScore 

    UpdatedPathsWithTerminalSymbol = set()
    UpdatedPathScore = {}
    
    # Then add in extensions of paths terminating in blanks
    for path in PathsWithTerminalBlank:
        i = 0
        for c in SymbolSet: # SymbolSet does not include blanks
            newpath = path+c
            UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition 
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[i+1]
            i+=1

               
        
    
    
    
    # First work on paths with terminal symbols
    for path in PathsWithTerminalSymbol:
    # Extend the path with
        i = 0
#         print('p',path[-1])
        for c in SymbolSet:  #every symbol other than blank SymbolSet does not include blanks
        # The final symbol is repeated, so this is the same symbol sequence
            
            if c == path[-1]:
                newpath = path
            else:
                newpath = path+c
            
            if newpath not in UpdatedPathsWithTerminalSymbol:
                UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
                UpdatedPathScore[newpath] = PathScore[path] * y[i+1]
            else:
                
                UpdatedPathScore[newpath] += PathScore[path] * y[i+1]
            
            i+=1

    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore


# In[10]:


# s = ['a','b','c','d']
# np.random.seed(0)
# a = np.random.rand(5,10,1)
# BeamSearch(s,a,2)

