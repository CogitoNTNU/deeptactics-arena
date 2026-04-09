class Node:
    def __init__(self,state, parent = None, action_from_parent = None, prior = 0.0, is_terminal = False, env = None, terminal_reward : float = 0.0):
           
        self.is_terminal = bool(is_terminal)
        self.terminal_reward = float(terminal_reward)
        

        #neural net stats
        self.prior = float(prior)
        self.state = state

        
        #mcts stats
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.parent = parent
        self.action_from_parent = action_from_parent
        
        #env
        self.env = env
        

    #funksjoner

    def get_mean_value(self):    #Regner Q(s,a), mean action value, bruker i mcsts
        if self.visit_count == 0:
            return 0.0
        return self.value_sum/self.visit_count

        
    

    def add_child(self, action, child_node):
        self.children[action] = child_node
        return child_node

    def is_expanded(self):
        return len(self.children) > 0
