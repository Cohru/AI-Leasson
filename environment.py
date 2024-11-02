
import numpy as np

# state space 
# (1,1),(2,1),(3,1),(4,1),(1,2),(3,2),(4,2),(1,3),(2,3),(3,3),(4,3)
#   0     1     2     3     4     5     6     7     8     9    10


# For coding simplicity, use the below one. 
# (1,1),(2,1),(3,1),(4,1),(1,2),(2,2),(3,2),(4,2),(1,3),(2,3),(3,3),(4,3)
#   0     1     2     3     4     5     6     7     8     9    10    11

# action space
# N,S,E,W
# 0,1,2,3


# getReward function
def getReward(state): 
    if (not(state==5 or state==7 or state==11)):
        R = -0.02
    elif state==5: 
        R = -.5 # you can try different values for this.  -0.1# math.nan 
    elif state==7:
        R = -1.
    else:
        R = 1.
    return R


# getNextstates
def getNextStates(s,a): 
    #x1,x2 = convertState1Dto2D(s)
    s2d = convertState1Dto2D(s)
    x1 = s2d[0]
    x2 = s2d[1]
    s2d_front = [x1,x2]
    s2d_left  = [x1,x2]
    s2d_right = [x1,x2]

    # if a == 0: # N 
    #     if (not(x2==3)):# or (x1==2 and x2==1))):
    #         s2d_front = [x1,x2+1]
    #     if (not(x1==1)): #or (x1==3 and x2==2))): 
    #         s2d_left = [x1-1,x2]
    #     if (not(x1==4)):# or (x1==1 and x2==2))):
    #         s2d_right = [x1+1,x2]
    
    # elif a == 1: # S 
    #     if (not(x2==1)):# or (x1==2 and x2==3))):
    #         s2d_front = [x1,x2-1]
    #     if (not(x1==4)):# or (x1==1 and x2==2))): 
    #         s2d_left = [x1+1,x2]
    #     if (not(x1==1)):# or (x1==3 and x2==2))):
    #         s2d_right = [x1-1,x2]
        
    # elif a == 2: # E
    #     if (not(x1==4)):# or (x1==1 and x2==2))):
    #         s2d_front = [x1+1,x2]
    #     if (not(x2==3)):# or (x1==2 and x2==1))): 
    #         s2d_left = [x1,x2+1]
    #     if (not(x2==1)):# or (x1==2 and x2==3))):
    #         s2d_right = [x1,x2-1]
    
    # else: # W
    #     if (not(x1==1)):# or (x1==3 and x2==2))):
    #         s2d_front = [x1-1,x2]
    #     if (not(x2==1)):# or (x1==2 and x2==3))): 
    #         s2d_left = [x1,x2-1]
    #     if (not(x2==3)):# or (x1==2 and x2==1))):
    #         s2d_right = [x1,x2+1]

    if a == 0: # N 
        if (not(x2==3 or (x1==2 and x2==1))):
            s2d_front = [x1,x2+1]
        if (not(x1==1 or (x1==3 and x2==2))): 
            s2d_left = [x1-1,x2]
        if (not(x1==4 or (x1==1 and x2==2))):
            s2d_right = [x1+1,x2]
    
    elif a == 1: # S 
        if (not(x2==1 or (x1==2 and x2==3))):
            s2d_front = [x1,x2-1]
        if (not(x1==4 or (x1==1 and x2==2))): 
            s2d_left = [x1+1,x2]
        if (not(x1==1 or (x1==3 and x2==2))):
            s2d_right = [x1-1,x2]
        
    elif a == 2: # E
        if (not(x1==4 or (x1==1 and x2==2))):
            s2d_front = [x1+1,x2]
        if (not(x2==3 or (x1==2 and x2==1))): 
            s2d_left = [x1,x2+1]
        if (not(x2==1 or (x1==2 and x2==3))):
            s2d_right = [x1,x2-1]
    
    else: # W
        if (not(x1==1 or (x1==3 and x2==2))):
            s2d_front = [x1-1,x2]
        if (not(x2==1 or (x1==2 and x2==3))): 
            s2d_left = [x1,x2-1]
        if (not(x2==3 or (x1==2 and x2==1))):
            s2d_right = [x1,x2+1]
    
    s1d_front = convertState2Dto1D(s2d_front) 
    s1d_left = convertState2Dto1D(s2d_left) 
    s1d_right = convertState2Dto1D(s2d_right) 
    
    n = np.random.rand()
    if (n < 0.8): 
        s_next = s1d_front
    elif (n>0.8 and n<0.9):
        s_next = s1d_left
    else:
        s_next = s1d_right
    
    #if s1d_front==5 or s1d_left==5 or s1d_right==5: 
    #    print('wrong')
    
    #if s==6:
    #print(s1d_front, s1d_left, s1d_right)
    #s_next = getNextState(s,a)
    #return x1,x2
    #return s1d_front, s1d_left, s1d_right
    return s_next


# conversion state1D to state2D 
def convertState1Dto2D(state_1D):
    if state_1D == 0 :
        #x1 = 1
        #x2 = 1
        state_2D = [1,1]
    elif state_1D == 1: 
        #x1 = 2
        #x2 = 1
        state_2D = [2,1]
    elif state_1D == 2: 
        #x1 = 3 
        #x2 = 1
        state_2D = [3,1]
    elif state_1D == 3: 
        #x1 = 4
        #x2 = 1
        state_2D = [4,1]
    elif state_1D == 4: 
        #x1 = 1
        #x2 = 2
        state_2D = [1,2]
    elif state_1D == 5: 
        #x1 = 2
        #x2 = 2
        state_2D = [2,2]
    elif state_1D == 6: 
        #x1 = 3 
        #x2 = 2
        state_2D = [3,2]
    elif state_1D == 7: 
        #x1 = 4
        #x2 = 2
        state_2D = [4,2]
    elif state_1D == 8: 
        #x1 = 1
        #x2 = 3
        state_2D = [1,3]
    elif state_1D == 9: 
        #x1 = 2 
        #x2 = 3
        state_2D = [2,3]
    elif state_1D == 10: 
        #x1 = 3 
        #x2 = 3 
        state_2D = [3,3]
    else: 
        #x1 = 4 
        #x2 = 3
        state_2D = [4,3]
    
    #return x1, x2 # state_2D
    return state_2D
    
# conversion state1D to state2D 
def convertState2Dto1D(state_2D):
    if state_2D == [1,1] :
        state_1D = 0
    elif state_2D == [2,1]: 
        state_1D = 1
    elif state_2D == [3,1]: 
        state_1D = 2
    elif state_2D == [4,1]: 
        state_1D = 3
    elif state_2D == [1,2]: 
        state_1D = 4
    elif state_2D == [2,2]: 
        state_1D = 5
    elif state_2D == [3,2]: 
        state_1D = 6
    elif state_2D == [4,2]: 
        state_1D = 7
    elif state_2D == [1,3]: 
        state_1D = 8
    elif state_2D == [2,3]: 
        state_1D = 9
    elif state_2D == [3,3]: 
        state_1D = 10
    else: 
        state_1D = 11
    
    return state_1D

def step(s,a):
    R = getReward(s)
    s_next = getNextStates(s,a)
    
    return s_next,R
