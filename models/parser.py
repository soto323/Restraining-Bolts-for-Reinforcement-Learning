class parser:
    def __init__(self):

# parser to parse the ltlf formula

        self.states = {"q1","q2","q3"}

        self.initialState  = "q1"
        self.finalState    = "q3"
        self.transitions()

    def transitions(self):

        self.transition = {

            ("q1","q1"):-1,
            ("q1","q2"):0.5,
            ("q2","q2"):0.1,
            ("q2","q3"):1
        }


        self.state_dict = {

            (False,False):"q1",
            (True,False):"q2",
            (True,True):"q3"

        }



    def reset(self):
        self.initialState = "q1"


    def trace(self,key,door):

        key_door = (key,door)
        intrinsic_reward  = 0

        if self.state_dict.get(key_door) == None:
            self.reset()
            done =False
        else:
            state  = self.state_dict[key_door]
            
            done  =  False


            if self.transition.get((self.initialState,state)) == None:
                self.reset()
                done = True

            elif self.transition.get((self.initialState,state)) != None:

                intrinsic_reward = self.transition[(self.initialState,state)]
                self.initialState = state 



            if done:
                intrinsic_reward = -10

        return intrinsic_reward,done


