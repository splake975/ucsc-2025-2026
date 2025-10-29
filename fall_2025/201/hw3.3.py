import numpy as np
import ast
import time

def precompute_cost1(r):
    # print(r)
    n = len(r)
    cost = np.zeros((n, n))
    for i in range(n): #starting index
        for j in range(i + 1, n): # ending index
            subtotal = 0
            for t in range(i + 1, j + 1):
                subtotal += r[t] * ((j + 1) - t)
                # print(subtotal,t,r[t],((j + 1) - t))
            cost[i, j] = subtotal
    return cost #cost[i,j] = sent after i, sent after j, cost of all requests i+1 to j
def precompute_cost(r):
    n = len(r)
    r = np.asarray(r, dtype=float)
    cost = np.zeros((n, n))

    # Precompute prefix sums
    prefix_r = np.cumsum(r)
    prefix_rt = np.cumsum(r * np.arange(n))

    # Compute cost[i, j] in O(1) per pair
    for i in range(n):#starting index
        for j in range(i + 1, n):# ending index
            sum_r = prefix_r[j] - prefix_r[i]
            sum_rt = prefix_rt[j] - prefix_rt[i]
            cost[i, j] = (j + 1) * sum_r - sum_rt

    return cost #cost[i,j] = sent after i, sent after j, cost of all requests i+1 to j




def optimal_broadcast2(R,K,cost:np.ndarray):
    dp = np.full((K,len(R)),np.inf)
    opt = np.zeros((K,len(R)))
    # opt = np.array([[np.array([]) for _ in range(len(R))] for _ in range(K)], dtype=object)
    # dp[0,0]=0
    for i in range(len(R)):
        dp[0,i]=cost[0,i]
        opt[0,i]=i
    
    for k in range(1,K): #k count
        for i in range(len(R)): # ending
            #j is the cut            
            # print(f"{[dp[k-1,j]+cost[j,i] for j in range(i)]=}")
            if [dp[k-1,j]+cost[j,i] for j in range(i)]:
                # dp[k,i] = np.inf
                best_j = 0
                for j in range(i):
                    if dp[k-1,j]+cost[j,i]<dp[k,i]: #better at j than j-1
                        dp[k,i] = dp[k-1,j]+cost[j,i]
                        best_j = j
                opt[k,i] = best_j
                
    return opt,dp

def optimal_broadcast3(R,K,cost:np.ndarray):
    dp = np.full((K,len(R)),np.inf)
    opt = np.zeros((K,len(R)))
    # opt = np.array([[np.array([]) for _ in range(len(R))] for _ in range(K)], dtype=object)
    # dp[0,0]=0
    for i in range(len(R)):
        dp[0,i]=cost[0,i]
        opt[0,i]=i
    
    for k in range(1,K): #k count
        for i in range(len(R)): # ending
            #j is the cut            
            if i == 0:
                continue  # no valid j < i
            vals = dp[k-1, :i] + cost[:i, i]
            best_j = np.argmin(vals)
            dp[k, i] = vals[best_j]
            opt[k, i] = best_j
                
    return opt,dp

def postprocess_opt(opt):
    cut = int(opt.shape[1]-1)
    # print(f"{cut,range(opt.shape[0])=}")
    ret = [opt.shape[1]-1]
    # print(f"{opt[0,cut]=}")
    r = reversed(range(1,opt.shape[0]))
    for i in r:
        # print(f"{int(opt[i,cut])=}")
        ret.append(int(opt[i,cut]))
        cut=int(opt[i,cut])
    ret = [r+1 for r in ret]
    return ret

def calc_score(start,end,schedule,sequence):
    if not schedule:
        return np.inf
    if schedule[-1] <= end-1:
        return np.inf
    i=0
    value=0
    for index in range(start,end):
        # print(index)
        if schedule[i]!=index:
            
            value+=sequence[index]*(schedule[i]-index)
            # print(f"{value=}")
        else:
            i+=1
            value+=sequence[index]*(schedule[i]-index)
            # print(f"{value=}")
    return value
    

if __name__ == "__main__":
    k=3
    r=[0, 3, 4, 0, 5, 2, 7]
    
    r=ast.literal_eval(input("r: "))
    k=int(input("K: "))
    
    start_time = time.time()
    
    
    cost = precompute_cost(r)
    print(cost)
    print()
    opt,dp=optimal_broadcast3(r,k,cost)
    print(*optimal_broadcast3(r,k,cost))
    opt_schedule = postprocess_opt(opt)
    opt_schedule.reverse()
    print(opt_schedule)
    print(calc_score(0,len(r),opt_schedule,r))
    print("time: ",time.time()-start_time)
