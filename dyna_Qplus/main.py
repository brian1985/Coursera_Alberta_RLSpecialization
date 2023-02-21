import numpy as np
from grid_Qcontrol_algorithm.main import EnvironmentGridWorldObstacles, fun_map, Episode


class AgentDynaQPlus:

    def __init__(self, num_states
                 , num_actions
                 , step_size=1
                 , epsilon=0.1
                 , discount=0.95
                 , kappa=0.05
                 , planning_steps=10
                 , random_seed=42
                 , plan_rand_seed=42):

        self.num_states = num_states
        self.num_actions = num_actions
        self.step_size = step_size
        self.gamma = discount
        self.kappa = kappa
        self.epsilon = epsilon
        self.planning_steps = planning_steps

        self.prior_state = None
        self.prior_action = None
        self.terminal = False

        self.q = np.zeros((self.num_states, self.num_actions))
        self.tau = np.zeros((self.num_states, self.num_actions))

        #  follow convention of course that diction is dict of states with values a dict over actions.
        #  {s: {action: (s_prime, reward)}
        self.model = {}

        #  for comparison debugging with Alberta Coursera course, I'll implement the same number generator
        self.rand_generator = np.random.RandomState(random_seed)  # use to break ties in argmax
        self.planning_rand_generator = np.random.RandomState(plan_rand_seed)  # use for other rand choice

    def argmax(self, q_values):
        current_max = float("-inf")
        ties = []

        for a, a_val in enumerate(q_values):
            if a_val > current_max:
                current_max = a_val
                ties = [a]

            elif a_val == current_max:
                ties.append(a)

        return self.rand_generator.choice(ties)

    def choose_action_egreedy(self, state):
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(4)
        else:
            action = self.argmax(self.q[state, :])

        return action

    def plan(self):
        for i in range(self.planning_steps):
            state = self.planning_rand_generator.choice(list(self.model.keys()))
            action = self.planning_rand_generator.choice(list(self.model[state].keys()))

            s_prime, reward = self.model[state][action]
            reward += self.kappa * np.sqrt(self.tau[state, action])

            max_q = 0 if s_prime == -1 else max(self.q[s_prime, :])
            self.q[state, action] += self.step_size * (reward + self.gamma * max_q - self.q[state, action])

    def update_model(self, s, a, s_prime, reward):
        if s in self.model.keys():
            self.model[s][a] = (s_prime, reward)
        else:
            self.model[s] = {a: (s_prime, 0)}
            for a_iter in range(self.num_actions):
                if a_iter != a:
                    self.model[s][a_iter] = (s, 0)

            self.model[s][a] = (s_prime, reward)

    def learn(self, s, reward, terminal):
        max_q = 0 if terminal else max(self.q[s, :])
        self.q[self.prior_state, self.prior_action] += self.step_size \
              * (reward + self.gamma * max_q - self.q[self.prior_state, self.prior_action])

    def start(self, state):
        self.prior_state = state
        self.prior_action = self.choose_action_egreedy(state)

        return self.prior_action

    def step(self, state, reward):
        self.tau += 1
        self.tau[self.prior_state, self.prior_action] = 0

        self.learn(state, reward, False)
        self.update_model(self.prior_state, self.prior_action, state, reward)
        self.plan()

        self.prior_state = state
        self.prior_action = self.choose_action_egreedy(state)

        return self.prior_action

    def terminal_learn(self, reward):
        self.tau += 1
        self.tau[self.prior_state, self.prior_action] = 0

        self.learn(self.prior_state, reward, True)
        self.update_model(self.prior_state, self.prior_action, -1, reward)
        self.plan()

def test_agent():
    actions = []
    agent_info = {
                  "epsilon": 0.1,
                  "step_size": 0.1,
                  "discount": 1.0,
                  "random_seed": 0,
                  "plan_rand_seed": 0}

    agent = AgentDynaQPlus(3, 4, **agent_info)

    agent.update_model(0, 2, 0, 1)
    agent.update_model(2, 0, 1, 1)
    agent.update_model(0, 3, 1, 2)
    agent.tau[0][0] += 1

    expected_model = {
        0: {
            0: (0, 0),
            1: (0, 0),
            2: (0, 1),
            3: (1, 2),
        },
        2: {
            0: (1, 1),
            1: (2, 0),
            2: (2, 0),
            3: (2, 0),
        },
    }
    assert agent.model == expected_model

    agent_info = {
                  "epsilon": 0.1,
                  "step_size": 0.1,
                  "discount": 1.0,
                  "kappa": 0.001,
                  "random_seed": 0,
                  "planning_steps": 4,
                  "plan_rand_seed": 0}

    agent = AgentDynaQPlus(3, 4, **agent_info)

    action = agent.start(0)  # state
    assert action == 1

    assert np.allclose(agent.tau, 0)
    assert np.allclose(agent.q, 0)
    assert agent.model == {}

    # ---------------
    # test agent step
    # ---------------

    action = agent.step(2,1)
    assert action == 3

    action = agent.step(1,0)
    assert action == 1

    expected_tau = np.array([
        [2, 1, 2, 2],
        [2, 2, 2, 2],
        [2, 2, 2, 0],
    ])
    print(agent.tau)
    assert np.all(agent.tau == expected_tau)

    expected_values = np.array([
        [0.0191, 0.271, 0.0, 0.0191],
        [0, 0, 0, 0],
        [0, 0.000183847763, 0.000424264069, 0],
    ])
    assert np.allclose(agent.q, expected_values)

    expected_model = {
        0: {
            1: (2, 1),
            0: (0, 0),
            2: (0, 0),
            3: (0, 0),
        },
        2: {
            3: (1, 0),
            0: (2, 0),
            1: (2, 0),
            2: (2, 0),
        },
    }
    assert agent.model == expected_model

    # --------------
    # test agent end
    # --------------
    agent.terminal_learn(1)

    expected_tau = np.array([
        [3, 2, 3, 3],
        [3, 0, 3, 3],
        [3, 3, 3, 1],
    ])
    assert np.all(agent.tau == expected_tau)

    expected_values = np.array([
        [0.0191, 0.344083848, 0, 0.0444632051],
        [0.0191732051, 0.19, 0, 0],
        [0, 0.000183847763, 0.000424264069, 0],
    ])
    assert np.allclose(agent.q, expected_values)

    expected_model = {0: {1: (2, 1), 0: (0, 0), 2: (0, 0), 3: (0, 0)}, 2: {3: (1, 0), 0: (2, 0), 1: (2, 0), 2: (2, 0)},
                      1: {1: (-1, 1), 0: (1, 0), 2: (1, 0), 3: (1, 0)}}
    assert agent.model == expected_model

    print("Assertions met")

#  test_agent()

TESTING = True

SIZE = (7, 12)
MAX_ROUNDS = 1000
NUM_EPISODES = 200

# agent seems to be able to learn well with wide range of step and learning rates.
agent_params = {
    "step_size": 1.0,
    "epsilon": 0.05,
    "discount_factor": 0.95
}

episode = Episode(SIZE, MAX_ROUNDS, agent_params=agent_params)

agent_params = {"epsilon": 0.1,
              "step_size": 0.55,
              "discount": 0.95,
              "kappa": 0.001,
              "planning_steps": 300,
              "random_seed": 0,
              "plan_rand_seed": 1}
episode.agent = AgentDynaQPlus(SIZE[0] * SIZE[1], 4, **agent_params)
#  overwrite the map to some default ones for testing
if TESTING:
    if SIZE == (7, 12):
        episode.env.map = fun_map.copy()

for i in range(NUM_EPISODES):
    final_round = episode.run_episode()
    if (i + 1) % 20 == 0:
        print(f"Episode {i + 1} ended on round {final_round}")

    episode.episode_restart()

print(f"Training Finished with Average rounds = {np.mean(episode.episode_tracker)}")

# build simple visual of optimal pathing after training
best_actions = np.argmax(episode.agent.q, axis=1)
best_actions = best_actions.reshape(-1, SIZE[1])
d = {0: "^", 1: "V", 2: "<", 3: ">"}
map = np.vectorize(d.get)(best_actions)
map = np.where(episode.env.map == 1, "######", map)
