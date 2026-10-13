# CartPole-v1 Solver using Double Deep Q-Network (DQN) & PyTorch

A clean, modular, and optimized implementation of a **Double Deep Q-Network (DQN)** agent designed to solve the classic **Gymnasium CartPole-v1** reinforcement learning environment.

---

## 🚀 Key Features
- **Double DQN Architecture:** Mitigates action-value overestimation for more stable and reliable training.

- **Experience Replay Memory:** Breaks correlation between consecutive samples using random mini-batch sampling.

- **Soft Target Updates (`TAU`):** Smoothly updates target network parameters rather than hard updates.


- **Command-Line Interface (CLI):** Easily configurable using `argparse` for training, evaluation, and rendering.
- **Logging & Checkpointing:** Automatically logs episode metrics to a CSV file (`cartpole_log.csv`) and saves the best model weights (`cartpole_dqn.pth`).

---

##  Tech Stack
- **Python**
- **PyTorch** (Deep learning framework)
- **Gymnasium** (Reinforcement learning toolkit)
- **NumPy & Pandas** (Data processing and logging)
- **Matplotlib** (Performance plotting)

---

## 📁 Project Structure
```text
cartpole-dqn-pytorch/
├── main.py             # Core DQN script & CLI handler
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── cartpole_dqn.pth    # Saved model weights (generated after training)

Usage Guide
​1. Train the Agent
​To train the DQN agent from scratch:python main.py --train

To train while rendering the environment window live: python main.py --train --render

2. Evaluate the Agent
​To test/evaluate a pre-trained model checkpoint with rendering
python main.py --eval --load cartpole_dqn.pth --render

📊 Training Performance
​The agent monitors the average reward over the last 100 episodes and automatically checkpoints the policy network whenever a new high score is reached.

