from typing import Optional
import numpy as np

                
class dp:
    def __init__(self,start,end,n) -> None:
        # self.subschedule = subschedule
        self.start=start
        self.end=end
        self.n=n
        self.opt = []
        self.wait = 0
    def score(self,opt):
        i=0
        wait=0
        for index in range(self.start,self.end):
            print(index)
            if opt[i]!=index:
                
                wait+=sequence[index]*(opt[i]-index)
                print(f"{wait=}")
            else:
                i+=1
                wait+=sequence[index]*(opt[i]-index)
                print(f"{wait=}")
        return wait
    def set_opt(self,opt):
        self.opt = opt
        i=0
        # temp = self.start
        for index in range(self.start,self.end):
            # while i<=len(self.opt):
            print(index)
            if self.opt[i]!=index:
                
                self.wait+=sequence[index]*(self.opt[i]-index)
                print(f"{self.wait=}")
            else:
                # temp = index
                i+=1
                self.wait+=sequence[index]*(self.opt[i]-index)
                print(f"{self.wait=}")
    def __str__(self) -> str:
        return self.__repr__()
    def __repr__(self) -> str:
        return "dp: "+str(self.start)+" "+str(self.end)+" "+str(self.n)






sequence=[0,1,2,3,4]
sequence=[0, 3, 4, 0, 5, 2, 7]
n=2
# for i in range(len(sequence)):
#     for j in range(1,len(sequence)-i+1):
#         # print(f"{j=}")
#         # print(sequence[i:i+j])
#         pass


start_list:dict[int,list[dp]]={i:[] for i in range(len(sequence))}
end_list:dict[int,list[dp]]={i:[] for i in range(len(sequence))}
en_list:dict[int,list[dp]]={en:[] for en in range(n)}
len_list:dict[int,list[dp]]={i:[] for i in range(len(sequence))}

dp_array:list[list[list[dp]]]=[]



# for en in n:
for en in range(0,n):
    dp_array.append([])
    for i in range(0,len(sequence)):
        dp_array[en].append([])
        for j in range(len(sequence)):
                # print(i,sequence[j:j+i])
                if len(sequence[j:j+i-1])==i+1:
                    dp_array[en][i].append(dp(j,j+i,en))
                    start_list[j].append(dp_array[en][i][-1])
                    end_list[j+i-1].append(start_list[j][-1])
                    en_list[en].append(start_list[j][-1])
                    len_list[i].append(start_list[j][-1])
                    # print(sequence[j:j+i-1])
                

# a=np.empty((len(sequence),len(sequence),n))

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


def optimal(start,end,n,a,sequence):
    if end-start<=n:
        a[start,end,n]=list(range(start+1,end+1))
        print(f"{(start,end,n)=}")
        print(f"{a[start,end,n]=}")
        return list(range(start+1,end+1))
    if a[start,end,n]:
        return a[start,end,n]
        return
    if n==0 or start==end:
        a[start,end,n]=None
        return None
    
    
    score = np.inf
    opt = []
    for first_half_n in range(n):
        for cut in range(start+1,end):
            calced_score = calc_score(start,end,optimal(start,cut,first_half_n,a,sequence),sequence)+calc_score(start,end,optimal(cut,end,n-first_half_n,a,sequence),sequence)
            print(f"{calced_score=}")
            if calced_score<score:
                score = calced_score
                opt = np.append(a[start,cut,first_half_n] , a[cut,end,n-first_half_n])
    
    a[start,end,n] = opt
    print(f"{(start,end,n)=}")
    print(f"{a[start,end,n]=}")
    return opt
    

def create_ordered_subproblems(sequence):
    ret = []
    for i in range(0,len(sequence)):
        # dp_array[en].append([])
        for j in range(len(sequence)):
            # print(i,sequence[j:j+i])
            if len(sequence[j:j+i+1])==i+1:
                
                ret.append([j,j+i+1])
                # print("appended")
                # ret.append(sequence[j:j+i-1])
                # print(sequence[j:j+i-1])
            # print(ret)
    return ret


if __name__ == "__main__":
    # test = dp(0,len(sequence),n)
    # test.set_opt([3,5,7])
    # print(test.score([3,5,7]))
    # print(test.wait)
    # print(len_list)
    # # len_list[6][0].set_opt([3,5,7])
    # # print(end_list[6][0].opt)
    # print(end_list)
    # print(dp_array)
    # # start_list[]
    sequence=[0, 3, 4, 0, 5, 2, 7]

    print(calc_score(0,len(sequence),[3,5,7],sequence))
    
    n=3
    a=np.empty((len(sequence),len(sequence)+1,n+1),dtype=object)
    # print(f"{a=}")
    subproblems = create_ordered_subproblems(sequence)
    print(f"{subproblems=}")
    input()
    for i in subproblems:
        # print(f"{(i[0],i[1],1)=}")
        optimal(i[0],i[1],1,a,sequence)
        # input()
        print("\n")
    # print(a[0,len(sequence)-1,3])
    print(a)
