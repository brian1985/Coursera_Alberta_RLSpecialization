"""
Q Learning algorithm

The Alberta Coursera course programming assignments felt very lacking
Filling in one or two lines for the algorithm wasn't provide clear evidence I understood it straight to finish
This code was written from scratch using only the algorithm steps as guide and then used some testing code
from the course to validate my results.

"""


import numpy as np

# some maps for testing
fun_map = np.array([
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
 [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
 [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
 [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

cliff = np.array([[0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,1,1,1,1,1,1,1,1,0]
                  ])

debug_map = np.array([[0,0,0]])


class EnvironmentGridWorldObstacles:
    """
    A grid Environment. Capable of different sizes as well as inserting obstacles to change up the map
    """
    def __init__(self, size=(3, 10), num_obstacles=9):
        self.OBSTACLE_VALUE = 1
        self.NUM_OBSTACLES = num_obstacles

        self.size = size # y,x format
        self.end = (self.size[0]-1, self.size[1]-1)

        self.terminal = False
        self.hit_obstacle = False
        self.actions = [0, 1, 2, 3]

        # init grid map with some obstacles
        self.map = np.zeros(self.size)
        self._build_obstacles()

        #  values that will be set on start of RL
        self.start_loc = None
        self.agent_loc = None
        self.flattened_loc = None  # flattened version of start 1-D

    def start(self, start_loc2D):
        self.start_loc = start_loc2D
        self.agent_loc = self.start_loc
        self.flattened_loc = self.set_flattened_loc()

        #  make sure the start and end don't contain obstacles
        self.map[self.start_loc] = 0
        self.map[self.end] = 0

    def _build_obstacles(self):
        for i in range(self.NUM_OBSTACLES):
            y = np.random.randint(0,self.size[0])
            x = np.random.randint(0,self.size[1])

            # in theory a random approach could lead to no valid path.
            # For this code, I'll just visually check there is one before running
            #  solution for pathing could be tree based maze solver
            self.map[y, x] = self.OBSTACLE_VALUE

    def update_environment(self, action):
        # action space is move up, down, right, left  = (0,1,2,3)
        y, x = self.agent_loc

        #action move agent. Off edge move implies no move.
        if action == 0:
            y = max(y-1, 0)

        elif action == 1:
            y = min(y+1,self.size[0]-1)

        elif action == 2:
            x = max(x-1,0)

        elif action == 3:
            x = min(x+1,self.size[1]-1)

        if self.map[(y,x)] == self.OBSTACLE_VALUE:
            self.hit_obstacle = True
        else:
            self.agent_loc = (y, x)

        self.set_flattened_loc()

    def set_flattened_loc(self):
        self.flattened_loc = self.size[1]*self.agent_loc[0] + self.agent_loc[1]

    def generate_reward(self):
        if self.hit_obstacle:
            return -100
        else:
            return -1

    def restart(self):
        self.agent_loc = self.start_loc
        self.flattened_loc = 0
        self.terminal = False

    def step(self, action):

        self.update_environment(action)
        reward = self.generate_reward()
        if self.agent_loc == self.end:
            self.terminal = True

        return reward, self.flattened_loc, self.terminal


class AgentQLearn_UDLR:
    """

    Agent that learns via Q learning with epsilon greedy action learning
    Agent is capable of 4 actions (up, down, left, right)
    To validate testing, I implemented the random number generator to match Coursera test values.

    """
    def __init__(self, size2d, num_actions=4, discount_factor = 1.0, step_size=1, epsilon=0.05):
        # q is a value function that rows are different actions, and columns are different complete states
        # Because map is static, the states are really locations of the agent, one per grid element

        self.q = np.zeros((num_actions, size2d[0]*size2d[1]))

        self.step_size = step_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_actions = num_actions

        self.prior_state = None
        self.prior_action = None

        self.rand_generator = np.random.RandomState(0)

    def learn(self, reward, new_location):
        max_q = max([self.q[a, new_location] for a in range(self.num_actions)])

        self.q[self.prior_action, self.prior_state] += self.step_size * \
                                    (reward + self.discount_factor * max_q - self.q[self.prior_action, self.prior_state])

    def argmax(self, state):
        max_ids = []
        current_max = 0

        for a in range(self.num_actions):
            if self.q[a, state] >= current_max:
                current_max = self.q[a, state]
                max_ids.append(a)
            else:
                pass

        if not max_ids:
            max_ids = list(range(self.num_actions))

        return np.random.choice(max_ids)

    def start(self, state):
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(state)

        self.prior_state = state
        self.prior_action = action

        return action

    def step(self, new_location, reward):

        self.learn(reward, new_location)

        # Explore with probability epsilon
        # greedy otherwise

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)

        else:
            action = self.argmax(new_location)

        # this action will be the next steps prior action
        self.prior_action = action

        # new_location will be the next steps prior state
        self.prior_state = new_location

        return action

    def terminal_learn(self, reward):
        self.q[self.prior_action, self.prior_state] += self.step_size * (reward - self.q[self.prior_action, self.prior_state])


class Episode:
    """

    Played around with building an Episode class for running a single Episode.
    Would contain tracking stats for an episode

    """
    def __init__(self, size, max_rounds, agent_params={}, env_params={}):
        self.MAX_ROUNDS = max_rounds

        self.env = EnvironmentGridWorldObstacles(size=size, num_obstacles=20, **env_params)
        self.agent = AgentQLearn_UDLR(size, len(self.env.actions), **agent_params)

        self.episode_tracker = []

    def run_episode(self, start_state=(0, 0)):
        action = self.agent.start(0)
        self.env.start(start_state)

        for i in range(self.MAX_ROUNDS):
            reward, new_state, terminal = self.env.step(action)

            if not terminal:
                action = self.agent.step(new_state, reward)
            else:
                self.agent.terminal_learn(reward)

                self.episode_tracker.append(i+1)
                return i+1

        self.episode_tracker.append(i+1)
        return i+1  #  if not terminal by max round, just return max round

    def episode_restart(self):
        self.env.restart()


# test Agent code for possible bugs
def testAgent():
    #Copied test values over from coursera course (Alberta RL- QLearn) for easy debugging instead of making up my own
    #actions are [0:3]= u,d,l,r

    #test values from start

    agent = AgentQLearn_UDLR((1, 3), 4, discount_factor=1.0, step_size=0.1, epsilon = 1.0)
    # agent.q[0,0] = -3
    action = agent.start(0)
    assert action == 1

    action = agent.step(1, 2)
    assert action == 3

    agent.terminal_learn(1)
    print(agent.q, agent.prior_action, agent.prior_state)

    for _ in range(100):
        agent.step(1, 2)
    print(agent.q, agent.prior_action, agent.prior_state)

# test EPisode code for bugs
def test_episode():
    agent_params = {
        "step_size": 0.1,
        "epsilon": 0.05,
        "discount_factor": 1.0
    }

    agent_params = {}

    #test params in agent
    a = AgentQLearn_UDLR((7, 12), 4, **agent_params)

    episode = Episode((7,12), 10)
    agent = episode.agent
    env = episode.env

    agent.start(0)
    env.start((0, 0))


if __name__ == "__main__":
    TESTING = True

    SIZE = (7, 12)
    MAX_ROUNDS = 1000
    NUM_EPISODES = 50

    # agent seems to be able to learn well with wide range of step and learning rates.
    agent_params = {
        "step_size": 1.0,
        "epsilon": 0.05,
        "discount_factor": 0.95
    }

    episode = Episode(SIZE, MAX_ROUNDS, agent_params=agent_params)

    #  overwrite the map to some default ones for testing
    if TESTING:
        if SIZE == (7, 12):
            episode.env.map = fun_map
        if SIZE == (4, 10):
            episode.env.map = cliff
        if SIZE == (1, 3):
            episode.env.map = debug_map

    for i in range(NUM_EPISODES):
        final_round = episode.run_episode()
        if (i + 1) % 50 == 0:
            print(f"Episode {i + 1} ended on round {final_round}")

        episode.episode_restart()

    print(f"Average rounds = {np.mean(episode.episode_tracker)}")

    # build simple visual of optimal pathing after training
    best_actions = np.argmax(episode.agent.q, axis=0)
    best_actions = best_actions.reshape(-1, SIZE[1])
    d = {0: "^", 1: "V", 2: "<", 3: ">"}
    map = np.vectorize(d.get)(best_actions)
    map = np.where(episode.env.map == 1, "######", map)