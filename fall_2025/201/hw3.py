from typing import Optional

def sub(sequence,start,end,n):
    subs=
    
    
    
class dp:
    
    def __init__(self,start:int,end:int,n:int) -> None:
        self.schedule:list[int]=[]
        self.n = n
        self.start = start
        self.end = end
        self.wait:Optional[int] = None
    @classmethod
    def total_schedule(cls,tschedule):
        cls.tschedule = tschedule
        
    def find_opt(self,subs):
        for split in range(self.end-self.start):
            for i in range(self.n):
                for sub in subs:
                    if sub.
    
    
    def valid_sub(self,sub,n):
        for sub in subs:
            if sub.start ==