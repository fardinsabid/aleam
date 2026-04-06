#!/usr/bin/env python3
"""
Reinforcement Learning example using Aleam true randomness.

This example demonstrates using Aleam's true random numbers for
exploration in reinforcement learning. True randomness ensures
genuine exploration of the state-action space without hidden patterns.
"""

import aleam as al
import numpy as np


class TrueRandomRL:
    """
    Reinforcement Learning agent using true randomness for exploration.
    
    This agent uses Aleam's true random numbers for epsilon-greedy
    exploration, ensuring no hidden patterns or periodic behavior
    in the exploration strategy.
    """
    
    def __init__(self, state_dim, action_dim, epsilon=0.1, learning_rate=0.1, gamma=0.99):
        """
        Initialize the RL agent.
        
        Args:
            state_dim: Dimension of state space (for discretization)
            action_dim: Number of possible actions
            epsilon: Initial exploration rate
            learning_rate: Alpha for Q-learning update
            gamma: Discount factor for future rewards
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Use true randomness for exploration
        self.rng = al.Aleam()
        self.ai = al.AIRandom()
        
        # Q-table for discrete state approximation
        self.q_table = {}
        
        # Track statistics
        self.total_steps = 0
        self.exploration_steps = 0
        self.exploitation_steps = 0
    
    def _get_state_key(self, state):
        """
        Discretize continuous state for Q-table.
        
        Args:
            state: Continuous state vector
            
        Returns:
            tuple: Discretized state key
        """
        # Discretize to 10 bins per dimension
        discretized = tuple(np.floor(state * 10).astype(int))
        return discretized
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy with true randomness.
        
        Args:
            state: Current state
            
        Returns:
            int: Selected action
        """
        state_key = self._get_state_key(state)
        self.total_steps += 1
        
        # Explore with true randomness
        if self.rng.random() < self.epsilon:
            self.exploration_steps += 1
            action = self.rng.randint(0, self.action_dim - 1)
            return action
        else:
            self.exploitation_steps += 1
            # Exploit using best action from Q-table
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_dim)
            return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-values using Q-learning.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        state_key = self._get_state_key(state)
        next_key = self._get_state_key(next_state)
        
        # Initialize Q-values if not present
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_dim)
        
        # Q-learning update
        best_next = np.max(self.q_table[next_key])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_error
    
    def decay_epsilon(self, decay=0.995, min_epsilon=0.01):
        """
        Decay exploration rate over time.
        
        Args:
            decay: Decay factor per episode
            min_epsilon: Minimum exploration rate
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay)
    
    def get_stats(self):
        """
        Get agent statistics.
        
        Returns:
            dict: Agent statistics
        """
        total = self.exploration_steps + self.exploitation_steps
        return {
            'total_steps': self.total_steps,
            'exploration_steps': self.exploration_steps,
            'exploitation_steps': self.exploitation_steps,
            'exploration_rate': self.exploration_steps / total if total > 0 else 0,
            'q_table_size': len(self.q_table),
            'current_epsilon': self.epsilon
        }


def mountain_car_environment():
    """
    Simple mountain car environment simulation.
    
    The goal is to drive an underpowered car to the top of a hill.
    Actions: 0 = push left, 1 = no push, 2 = push right
    """
    
    class MountainCar:
        def __init__(self):
            self.min_position = -1.2
            self.max_position = 0.6
            self.goal_position = 0.5
            self.reset()
        
        def reset(self):
            """Reset environment to random starting position."""
            self.position = self.rng.uniform(-0.6, -0.4)
            self.velocity = 0.0
            return np.array([self.position, self.velocity])
        
        def step(self, action):
            """
            Take a step in the environment.
            
            Args:
                action: 0 (left), 1 (nothing), or 2 (right)
                
            Returns:
                tuple: (next_state, reward, done)
            """
            # Action force
            force = (action - 1) * 0.001
            
            # Update velocity
            self.velocity += force - 0.0025 * np.cos(3 * self.position)
            self.velocity = np.clip(self.velocity, -0.07, 0.07)
            
            # Update position
            self.position += self.velocity
            self.position = np.clip(self.position, self.min_position, self.max_position)
            
            # Penalize for being on the left side
            reward = -1.0
            
            # Check if goal reached
            done = self.position >= self.goal_position
            if done:
                reward = 100.0
            
            return np.array([self.position, self.velocity]), reward, done
        
        def render(self):
            """Simple text-based rendering."""
            bar_length = 50
            pos_normalized = (self.position - self.min_position) / (self.max_position - self.min_position)
            pos_idx = int(pos_normalized * bar_length)
            bar = ['-'] * bar_length
            bar[pos_idx] = 'C'  # Car
            bar[int(bar_length * 0.8)] = 'G'  # Goal
            print(''.join(bar))
    
    env = MountainCar()
    env.rng = al.Aleam()  # Use true randomness for environment
    return env


def main():
    print("=" * 70)
    print("Aleam - Reinforcement Learning Example")
    print("=" * 70)
    
    # Create environment and agent
    env = mountain_car_environment()
    agent = TrueRandomRL(state_dim=2, action_dim=3, epsilon=0.1, learning_rate=0.1)
    
    print("\n🎯 Training Mountain Car with True Random Exploration")
    print("   (Goal: Reach the top of the hill at position 0.5)")
    print("   True randomness enables genuine exploration without hidden patterns\n")
    
    episodes = 500
    total_rewards = []
    best_reward = -float('inf')
    
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
        
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(total_rewards[-50:])
            stats = agent.get_stats()
            print(f"  Episode {episode+1:4d}: avg reward = {avg_reward:6.1f}, "
                  f"epsilon = {agent.epsilon:.3f}, "
                  f"exploration = {stats['exploration_rate']:.1%}")
    
    print("\n📊 Training Summary:")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Best episode reward: {best_reward:.1f}")
    print(f"  Average reward (last 100 episodes): {np.mean(total_rewards[-100:]):.1f}")
    
    stats = agent.get_stats()
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Exploration steps: {stats['exploration_steps']} ({stats['exploration_rate']:.1%})")
    print(f"  Q-table size: {stats['q_table_size']}")
    
    print("\n" + "=" * 70)
    print("✅ Reinforcement Learning demo complete")
    print("   True randomness enables genuine exploration")
    print("=" * 70)


if __name__ == "__main__":
    main()