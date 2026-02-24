class Solution:
    def hasAllCodes(self, s: str, k: int) -> bool:
        maincheck = self.gen(k)
        # for i in range(k):
        #     maincheck+=self.gen(i+1)
        
        
        print(maincheck)
        p = 2**k
        for i in range(p):
            if maincheck[i] in s:
                continue 
            return False
        return True
            # for j in range(s.__len__() + k):
        
        
    def gen(self,k:int)->list[str]:
        check:list[str]=["0","1"]
        for _ in range(k-1):
            tlen=len(check)
            check = check+check
            for i in range(tlen):
                check[i]+="0"
                check[tlen+i]+="1"
        return check
        
if __name__ == "__main__":
    s=Solution()
    s.hasAllCodes("",3)