import numpy as np

def precompute_cost(r):
    print(r)
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


def optimal_broadcast(R, K):
    n = len(R)
    cost = precompute_cost(R)
    dp = np.full((n + 1, K + 1), np.inf)
    prev = np.full((n + 1, K + 1), -1, dtype=int)

    dp[0][0] = 0

    for k in range(1, K + 1):
        for i in range(1, n + 1):
            for p in range(i):
                new_cost = dp[p][k - 1] + cost[p][i - 1]
                if new_cost < dp[i][k]:
                    dp[i][k] = new_cost
                    prev[i][k] = p

    # Reconstruct schedule
    schedule = []
    i, k = n, K
    while k > 0 and prev[i][k] != -1:
        p = prev[i][k]
        schedule.append(i)  # broadcast at i
        i, k = p, k - 1

    schedule = sorted(schedule)
    print(f"{dp=}")
    return dp[n][K], schedule

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
                # dp[k,i] = min(*[dp[k-1,j]+cost[j,i] for j in range(i)],np.inf)
            # dp[k,i]=min(,)
    # print(f"{dp2=}")
    # print(f"{opt=}")
    return opt,dp

def postprocess_opt(opt):
    cut = int(opt.shape[1]-1)
    print(f"{cut,range(opt.shape[0])=}")
    ret = [opt.shape[1]-1]
    # print(f"{opt[0,cut]=}")
    r = reversed(range(1,opt.shape[0]))
    for i in r:
        print(f"{int(opt[i,cut])=}")
        ret.append(int(opt[i,cut]))
        cut=int(opt[i,cut])
    ret = [r+1 for r in ret]
    return ret


    

if __name__ == "__main__":
    k=3
    r=[0, 3, 4, 0, 5, 2, 7]
    cost = precompute_cost(r)
    print(cost)
    print()
    opt,dp=optimal_broadcast2(r,k,cost)
    print(*optimal_broadcast2(r,k,cost))
    opt_schedule = postprocess_opt(opt)
    print(opt_schedule)
    
