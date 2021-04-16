class PIDcontrol():
    def __init__(self,Kp,Ti,Td,stable,expect):
        self.controlp=[Kp,Ti,Td]
        self.errorn=0
        self.errorn1=0
        self.errorn2=0
        self.deltau=0
        self.controlu=stable
        self.expect=expect
        self.controlseries=[]
    def control(self,input):
        self.errorn2=self.errorn1
        self.errorn1=self.errorn
        self.errorn=self.expect-input

        self.deltau=self.controlp[0]*(self.errorn-self.errorn1+self.errorn/self.controlp[1]+
                    self.controlp[2]*(self.errorn+self.errorn2-2*self.errorn1))
        self.controlu+=self.deltau
        self.controlseries.extend([self.controlu])
        return self.controlu
    
    def modify(self,expect):
        self.expect=expect