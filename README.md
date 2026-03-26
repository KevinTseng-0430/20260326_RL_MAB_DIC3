# Multi-Armed Bandit (MAB) Strategy Dashboard

Welcome to the **Multi-Armed Bandit Strategy Dashboard**, an interactive web application designed to demonstrate the critical trade-offs between exploration and exploitation in decision-making and optimization. 

This project was built following the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology to ensure a structured, business-oriented approach to comparing traditional A/B testing with advanced Multi-Armed Bandit algorithms.
<img width="1186" height="318" alt="image" src="https://github.com/user-attachments/assets/ac595f81-0b53-40f7-9b45-70eb4685481d" />
<img width="1176" height="623" alt="image" src="https://github.com/user-attachments/assets/2f1f3c6d-3ecb-4c82-88f2-10761288a9bc" />
<img width="1168" height="632" alt="image" src="https://github.com/user-attachments/assets/6c6448b3-ddae-4871-97f9-1c6c8a72d87b" />

Live Demo: https://kevintseng-0430-20260326-rl-mab-dic3-mab-app-etwz7c.streamlit.app/

## CRISP-DM Process Overview

### 1. Business Understanding
**Objective:** Maximize the expected return from a total budget of $10,000 allocated across three options (Bandit A, B, and C) with unknown true success probabilities.
**Problem:** Traditional A/B Testing often wastes significant resources (budget) on inferior options during its strict "exploration phase" and completely ignores other potential options (Bandit C). 
**Goal:** Prove that modern Bandit algorithms (like Upper Confidence Bound or Thompson Sampling) can yield higher total rewards and lower cumulative regret by dynamically balancing exploration and exploitation.

### 2. Data Understanding
In this reinforcement learning simulation, there is no pre-existing static dataset. Instead, the "data" is generated continuously through interactions with the environment:
* **True Means (Probabilities):** Defaulting to Bandit A (0.8), Bandit B (0.7), and Bandit C (0.5).
* **Rewards:** Each interaction (pull) yields a binary reward (1 or 0) drawn from a Binomial distribution based on the arm's true probability.

### 3. Data Preparation
To ensure the simulation is statistically robust and fair:
* We structured a **Monte Carlo Simulation** environment.
* We configured the system to perform $N$ independent runs (e.g., 30 runs) for each algorithm over a $10,000$-step budget.
* The rewards and "arm pull" counts were aggregated and averaged to eliminate random noise and produce reliable, smooth performance curves.

### 4. Modeling
We implemented and simulated six distinct strategies to solve the Explore-Exploit dilemma:
1. **Classic A/B Testing:** Static exploration (splitting 20% of the budget equally between A and B) followed by pure exploitation.
2. **Optimistic Initial Values:** Implicit exploration driven by inflated initial expectations, causing early disappointment and subsequent switching.
3. **$\epsilon$-Greedy:** Continues to explore randomly at a fixed probability ($\epsilon$).
4. **Softmax (Boltzmann Distribution):** Probabilistic exploration guided by a temperature parameter ($\tau$).
5. **Upper Confidence Bound (UCB):** Deterministic exploration driven by calculating the upper bound of uncertainty for each option.
6. **Thompson Sampling:** A Bayesian approach that updates probability distributions (Beta distribution) for each arm dynamically.

### 5. Evaluation
We evaluated the models based on three key performance indicators (KPIs):
* **Total Expected Reward:** How much money did the strategy make?
* **Cumulative Regret:** How much money did the strategy *lose* compared to an omniscient optimal strategy (always picking Bandit A)?
* **Arm Allocation Efficiency:** Did the algorithm minimize pulls on the sub-optimal Bandits B and C?

*Result:* As visualized in the dashboard, Thompson Sampling and UCB drastically outperform standard A/B Testing by reducing regret to a flat plain minimizing wasted budget.

### 6. Deployment
The final evaluation and model interactions were deployed as an interactive Web Application using **Streamlit**.
* **Interactivity:** Users can modify budgets, true probabilities, and algorithm hyperparameters (like $\epsilon$, temperature, and confidence bounds) in real-time.
* **Visualization:** The app features real-time plotting of Cumulative Regret lines and Arm Pull Allocation bar charts.
* **Accessibility:** The dashboard serves as an educational and analytical tool for stakeholders to quickly grasp the value of Bandit algorithms over traditional A/B testing.

---

## How to Run the Dashboard

### Prerequisites
Make sure you have Python installed along with the required libraries.
```bash
pip install streamlit pandas numpy
```

### Running the App
Navigate to the project directory and run the following command in your terminal:
```bash
streamlit run mab_app.py
```
The application will launch in your default web browser at `http://localhost:8501`.
