class Solution:
    def hasAllCodes(self, s: str, k: int) -> bool:
        tree = {i:False for i in self.gen(k)}
        for i in range(len(s)):
            test = s[i:i+k]
            print(f"{test=}")
            if len(test)!=k: continue 
            else: pass
            tree[test]=True
        for i in tree.keys():
            if tree[i]:
                continue
            return False
        return True
            
            
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
    print(s.hasAllCodes("00110110",2))