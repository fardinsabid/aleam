"""
Reinforcement Learning example using Aleam true randomness.
"""

import aleam as al
import numpy as np


class TrueRandomRL:
    """
    Reinforcement Learning agent using true randomness for exploration.
    """
    
    def __init__(self, state_dim, action_dim, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.rng = al.Aleam()
        self.ai = al.AIRandom()
        
        # Simple Q-table (for discrete state approximation)
        self.q_table = {}
    
    def _get_state_key(self, state):
        """Discretize continuous state for Q-table"""
        # Simple discretization - in practice use neural networks
        discretized = tuple(np.floor(state * 10).astype(int))
        return discretized
    
    def select_action(self, state):
        """Select action using epsilon-greedy with true randomness"""
        state_key = self._get_state_key(state)
        
        if self.rng.random() < self.epsilon:
            # Explore with true randomness
            action = self.rng.randint(0, self.action_dim - 1)
            return action
        else:
            # Exploit using best action from Q-table
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_dim)
            return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, alpha=0.1, gamma=0.99):
        """Update Q-values"""
        state_key = self._get_state_key(state)
        next_key = self._get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_dim)
        
        # Q-learning update
        best_next = np.max(self.q_table[next_key])
        td_target = reward + gamma * best_next
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += alpha * td_error
    
    def decay_epsilon(self, decay=0.995):
        """Decay exploration rate"""
        self.epsilon *= decay


def mountain_car_environment():
    """Simple mountain car environment simulation"""
    
    class MountainCar:
        def __init__(self):
            self.position = 0.0
            self.velocity = 0.0
            self.min_position = -1.2
            self.max_position = 0.6
            self.goal_position = 0.5
        
        def reset(self):
            self.position = self.rng.uniform(-0.6, -0.4)
            self.velocity = 0.0
            return np.array([self.position, self.velocity])
        
        def step(self, action):
            # Action: 0 = left, 1 = nothing, 2 = right
            force = (action - 1) * 0.001
            self.velocity += force - 0.0025 * np.cos(3 * self.position)
            if self.velocity > 0.07:
                self.velocity = 0.07
            if self.velocity < -0.07:
                self.velocity = -0.07
            
            self.position += self.velocity
            
            if self.position < self.min_position:
                self.position = self.min_position
                self.velocity = 0.0
            if self.position > self.max_position:
                self.position = self.max_position
            
            reward = -1.0
            done = self.position >= self.goal_position
            
            if done:
                reward = 100.0
            
            return np.array([self.position, self.velocity]), reward, done
    
    env = MountainCar()
    env.rng = al.Aleam()  # Add true randomness to environment
    return env


def main():
    print("=" * 70)
    print("ALEAM - Reinforcement Learning Example")
    print("=" * 70)
    
    # Create environment and agent
    env = mountain_car_environment()
    agent = TrueRandomRL(state_dim=2, action_dim=3, epsilon=0.1)
    
    print("\n🎯 Training Mountain Car with True Random Exploration")
    print("   (Goal: Reach the top of the hill at position 0.5)")
    
    episodes = 500
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:
            # Select action with true randomness
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Update agent
            agent.update(state, action, reward, next_state)
            
            state = next_state
            steps += 1
        
        total_rewards.append(episode_reward)
        agent.decay_epsilon()
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(total_rewards[-50:])
            print(f"  Episode {episode+1:4d}: avg reward = {avg_reward:.1f}, epsilon = {agent.epsilon:.3f}")
    
    print("\n📊 Training Summary:")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Best episode reward: {max(total_rewards):.1f}")
    print(f"  Average reward (last 100 episodes): {np.mean(total_rewards[-100:]):.1f}")
    
    print("\n" + "=" * 70)
    print("✅ Reinforcement Learning demo complete")
    print("   True randomness enables genuine exploration")
    print("=" * 70)


if __name__ == "__main__":
    main()