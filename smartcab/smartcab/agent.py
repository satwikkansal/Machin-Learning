import random
import numpy
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.q_table = {}
        self.reward = 0
        self.alpha = 0.9
        self.gamma = 0.1
        self.epsilon = 0.2

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.q_table = {}
        self.reward = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        #(state, action) pairs of the state
        self.state = inputs
        self.state['next_waypoint'] = self.next_waypoint
        self.state = tuple(self.state.values())

        if not self.state in self.q_table.keys():
            self.q_table[self.state] = {ac:0 for ac in self.env.valid_actions}

        
        # TODO: Select action according to your policy
        #We'll choose maximum q_value action or random action depending on the comparison result from epsilon
        max_q = 0
        action = random.choice(self.env.valid_actions)

        if numpy.random.random()>self.epsilon:            
            for act,val in self.q_table[self.state].iteritems():
                if val > max_q:
                    action = act


        # Execute action and get reward
        reward = self.env.act(self, action)

        #get the next state information
        new_inputs = self.env.sense(self)
        new_waypoint = self.planner.next_waypoint()
        self.next_state = new_inputs
        self.next_state['next_waypoint'] = new_waypoint
        self.next_state = tuple(self.next_state.values())

        #check if next_state has q_values already
        if self.next_state not in self.q_table:
            self.q_table[self.next_state] = {ac:0 for ac in self.env.valid_actions}

        # TODO: Learn policy based on state, action, reward
        old_q_value = self.q_table[self.state][action]
        #maximum q_value for next_state actions
        next_max = max(self.q_table[self.next_state].values())
        new_q_value = (1 - self.alpha)*old_q_value + self.alpha*(reward + self.gamma*next_max)
        self.q_table[self.state][action] = new_q_value
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    #running 10 trials without exploration to see whether we've learned enough or not.
    a.epsilon = 0.0
    sim.run(n_trials=20)


if __name__ == '__main__':
    run()
