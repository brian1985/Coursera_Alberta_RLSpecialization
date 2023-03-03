"""

For this example rather than build an environment I want to use openAI GYM package (gymnasium)

"""

from TileCoder.tilingSutton import IHT, tiles
import gymnasium as gym
import numpy as np

class Tiler:
    def __init__(self, size, num_tiles=8, num_tilings=8):
        self.size = size
        self.iht = IHT(self.size)
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings

    def get_tiles(self, state,  scale_p=(-1.2, 0.5), scale_v=(-0.07, 0.07)):
        pos, vel = state

        pos_scaled = (pos - scale_p[0]) / (-scale_p[0] + scale_p[1]) * self.num_tiles
        vel_scaled = (vel - scale_v[0]) / (-scale_v[0] + scale_v[1]) * self.num_tiles

        return tiles(self.iht, self.num_tilings, [pos_scaled, vel_scaled])


class AgentLinearSarsaTiling:
    def __init__(self, num_actions, epsilon=0.05, gamma=1.0, alpha = 0.5):
        self.tc = Tiler(4096)

        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha / self.tc.num_tilings

        self.p_range = (-1.2,0.5)
        self.v_range = (-0.7, 0.7)

        self.num_actions = num_actions

        #  need to track prior action/state pair for update step
        self.last_action = None
        self.last_tiles = None


        #  tiling approach to the state space. The size should be a value greater than num tiles * num tilings when possible
        self.w = np.zeros((num_actions, self.tc.size))

    def learn(self, reward, q_value, terminal):
        lt = self.last_tiles
        la = self.last_action

        if terminal:
            self.w[la, lt] += self.alpha * (reward - np.sum(self.w[la, lt]))
        else:
            self.w[la, lt] += self.alpha * (reward + self.gamma * q_value - np.sum(self.w[la, lt]))

    def get_q_values(self, tiles):
        """
        some agent learners it is simple to get the action value. Others require more sophisticated approaches,
        especially with the use case of linear approximations when the value is a function of many weights
        This gives space for defining how to get the q values from actions
        """
        q_values = []

        for a in range(self.num_actions):
            q_values.append(np.sum(self.w[a, tiles]))

        return q_values

    def get_action_egreedy(self, tiles):
        def argmax(q_values):
            ties = []
            current_max = q_values[0]

            for idx, q in enumerate(q_values):
                if q > current_max:
                    current_max = q
                    ties = [idx]
                elif q == current_max:
                    ties.append(idx)
            return np.random.choice(ties)

        q_values = self.get_q_values(tiles)

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = argmax(q_values)

        return action, q_values[action]

    def start(self, state):
        """
        Get the first action based on initial state
        """

        tiles = self.tc.get_tiles(state, self.p_range, self.v_range)
        self.last_action, _ = self.get_action_egreedy(tiles)
        self.last_tiles = tiles.copy()

        return self.last_action

    def step(self, reward, state):
        tiles = self.tc.get_tiles(state, self.p_range, self.v_range)
        action, q_value = self.get_action_egreedy(tiles)

        self.learn(reward, q_value, False)

        self.last_action = action
        self.last_tiles = tiles

        return action

    def end(self, reward):
        self.learn(reward, 0, True)

    def play(self, state):
        """
        Get an action without learning. Used after training to play using trained agent
        """
        tiles = self.tc.get_tiles(state, self.p_range, self.v_range)
        action, _ = self.get_action_egreedy(tiles)
        return action


def test_Tiler():
    import itertools
    # create a range of positions and velocities to test
    # then test every element in the cross-product between these lists
    pos_tests = np.linspace(-1.2, 0.5, num=5)
    vel_tests = np.linspace(-0.07, 0.07, num=5)
    tests = list(itertools.product(pos_tests, vel_tests))

    mctc = Tiler(size=1024, num_tilings=8, num_tiles=2)

    t = []
    for test in tests:
        position, velocity = test
        tiles = mctc.get_tiles((position,velocity))
        t.append(tiles)

    expected = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 8, 3, 9, 10, 6, 11],
        [12, 13, 8, 14, 9, 10, 15, 11],
        [12, 13, 16, 14, 17, 18, 15, 19],
        [20, 21, 16, 22, 17, 18, 23, 19],
        [0, 1, 2, 3, 24, 25, 26, 27],
        [0, 1, 8, 3, 28, 29, 26, 30],
        [12, 13, 8, 14, 28, 29, 31, 30],
        [12, 13, 16, 14, 32, 33, 31, 34],
        [20, 21, 16, 22, 32, 33, 35, 34],
        [36, 37, 38, 39, 24, 25, 26, 27],
        [36, 37, 40, 39, 28, 29, 26, 30],
        [41, 42, 40, 43, 28, 29, 31, 30],
        [41, 42, 44, 43, 32, 33, 31, 34],
        [45, 46, 44, 47, 32, 33, 35, 34],
        [36, 37, 38, 39, 48, 49, 50, 51],
        [36, 37, 40, 39, 52, 53, 50, 54],
        [41, 42, 40, 43, 52, 53, 55, 54],
        [41, 42, 44, 43, 56, 57, 55, 58],
        [45, 46, 44, 47, 56, 57, 59, 58],
        [60, 61, 62, 63, 48, 49, 50, 51],
        [60, 61, 64, 63, 52, 53, 50, 54],
        [65, 66, 64, 67, 52, 53, 55, 54],
        [65, 66, 68, 67, 56, 57, 55, 58],
        [69, 70, 68, 71, 56, 57, 59, 58],
    ]
    assert np.all(expected == np.array(t))
#  test_Tiler()
print("Tile testing complete!")

def test_start_method():
    # -----------
    # Tested Cell
    # -----------
    # The contents of the cell will be tested by the autograder.
    # If they do not pass here, they will not pass there.

    np.random.seed(0)

    agent = AgentLinearSarsaTiling(3, epsilon=0.1)

    agent.w = np.array([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])])

    action_distribution = np.zeros(3)
    for i in range(1000):
        chosen_action, action_value = agent.get_action_egreedy(np.array([0, 1]))
        action_distribution[chosen_action] += 1

    print("action distribution:", action_distribution)
    # notice that the two non-greedy actions are roughly uniformly distributed
    assert np.all(action_distribution == [29, 35, 936])

    agent = AgentLinearSarsaTiling(3, epsilon=0.0)
    agent.w = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    chosen_action, action_value = agent.get_action_egreedy([0, 1])

    assert chosen_action == 2
    assert action_value == 15

    # -----------
    # test update
    # -----------
    agent = AgentLinearSarsaTiling(3, epsilon=0.1)

    agent.start((0.1, 0.3))
    agent.step(1, (0.02, 0.1))

    assert np.all(agent.w[0, 0:8] == 0.0625)
    assert np.all(agent.w[1:] == 0)

#  test_start_method()


if __name__ == "__main__":

    def train():
        agent = AgentLinearSarsaTiling(3)
        env = gym.make("MountainCar-v0", render_mode="human")

        len_episode = 2000
        num_episodes = 200

        for epi in range(num_episodes):
            print(f"starting episode {epi}")
            state, _ = env.reset()
            action = agent.start(state)
            for i in range(len_episode):
                state, reward, terminal, truncated, info = env.step(action)
                if terminal:
                    agent.end(reward)
                    break
                else:
                    action = agent.step(reward, state)

        env.close()
        return agent

    print("begin training")
    a = train()

    a.epsilon = 0.01
    print("Training Finished")
    e = gym.make("MountainCar-v0", render_mode="human")
    state, _ = e.reset()
    reward = 0

    for _ in range(1000):
        action = a.play(state)
        print(action, state)
        state, reward, terminal, truncated, info = e.step(action)
        if terminal:
            break
    print("Fin - testing")
    e.close()

