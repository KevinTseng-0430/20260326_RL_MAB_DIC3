import streamlit as st
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="MAB Strategy Dashboard", page_icon="🎰", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Premium UI ---
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F8F9FA; }
    /* Headers */
    h1, h2, h3 { color: #2C3E50; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    /* Top Metric Cards Customization */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #EAECEF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        color: #1F77B4;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        color: #7F8C8D;
        font-weight: 500;
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; padding-top: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 55px; white-space: pre-wrap; background-color: #E9ECEF;
        border-radius: 8px 8px 0px 0px; padding: 0 24px; font-weight: 600;
        color: #495057; border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF; color: #1F77B4; border: 1px solid #DEE2E6;
        border-bottom: 2px solid transparent;
    }
    .dataframe-container { border-radius: 12px; overflow: hidden; border: 1px solid #EAECEF; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8634/8634080.png", width=80) 
    st.title("⚙️ Control Panel")
    st.markdown("Adjust simulation parameters to observe algorithm performance in real-time.")
    
    st.divider()
    
    st.subheader("Global Settings")
    budget = st.number_input("🎯 Total Budget (Steps)", min_value=1000, max_value=50000, value=10000, step=1000)
    runs = st.slider("🔄 Monte Carlo Runs", 10, 100, 30, step=10, help="More runs reduce noise but increase compute time.")
    
    st.divider()
    
    st.subheader("🎰 True Probabilities")
    mean_A = st.slider("Bandit A (Optimal)", 0.0, 1.0, 0.80, 0.05)
    mean_B = st.slider("Bandit B", 0.0, 1.0, 0.70, 0.05)
    mean_C = st.slider("Bandit C", 0.0, 1.0, 0.50, 0.05)
    true_means = np.array([mean_A, mean_B, mean_C])

    st.divider()
    
    st.subheader("🎛️ Algorithm Hyperparameters")
    epsilon = st.slider("ε-Greedy (Epsilon)", 0.0, 1.0, 0.10, 0.01, help="Probability of random exploration.")
    softmax_tau = st.slider("Softmax (Temperature)", 0.01, 1.0, 0.10, 0.01, help="Higher tau = more random exploration.")
    ucb_c = st.slider("UCB (Confidence 'c')", 0.1, 5.0, 2.0, 0.1, help="Exploration weight parameter.")
    
    st.divider()
    st.caption("Developed dynamically for MAB analysis.")

# --- Core Simulation ---
@st.cache_data(show_spinner=False)
def run_mab_simulation(budget, runs, means_tuple, eps, tau, c):
    true_means = np.array(means_tuple)
    k = len(true_means)
    methods = ["A/B Test", "Optimistic", "ε-Greedy", "Softmax", "UCB", "Thompson"]
    
    avg_rewards = {m: np.zeros(budget) for m in methods}
    avg_pulls = {m: np.zeros(k) for m in methods}
    
    for r in range(runs):
        for m in methods:
            Q = np.zeros(k)
            N = np.zeros(k)
            rewards = np.zeros(budget)
            
            if m == "Optimistic":
                Q = np.full(k, 5.0)
                alpha = 0.1
            elif m == "Thompson":
                alpha_beta = np.ones((k, 2))
                
            best_ab_action = 0
            
            for t in range(budget):
                # 1. Action
                if m == "A/B Test":
                    if t < int(budget * 0.2): # 20% of budget
                        action = t % 2
                    else:
                        if t == int(budget * 0.2):
                            best_ab_action = np.argmax(Q[:2])
                        action = best_ab_action
                elif m == "Optimistic":
                    action = np.argmax(Q)
                elif m == "ε-Greedy":
                    if np.random.rand() < eps:
                        action = np.random.randint(k)
                    else:
                        max_val = np.max(Q)
                        action = np.random.choice(np.where(Q == max_val)[0])
                elif m == "Softmax":
                    max_q = np.max(Q)
                    exp_q = np.exp((Q - max_q) / tau)
                    probs = exp_q / np.sum(exp_q)
                    action = np.random.choice(k, p=probs)
                elif m == "UCB":
                    if t < k:
                        action = t
                    else:
                        ucb_values = Q + c * np.sqrt(np.log(t+1) / N)
                        action = np.argmax(ucb_values)
                elif m == "Thompson":
                    samples = [np.random.beta(alpha_beta[i,0], alpha_beta[i,1]) for i in range(k)]
                    action = np.argmax(samples)
                
                # 2. Reward
                reward = np.random.binomial(1, true_means[action])
                rewards[t] = reward
                
                # 3. Update
                N[action] += 1
                if m == "Optimistic":
                    Q[action] += alpha * (reward - Q[action])
                elif m == "Thompson":
                    alpha_beta[action, 0] += reward
                    alpha_beta[action, 1] += 1 - reward
                else:
                    Q[action] += (1 / N[action]) * (reward - Q[action])
            
            avg_rewards[m] += rewards
            avg_pulls[m] += N
            
    for m in methods:
        avg_rewards[m] /= runs
        avg_pulls[m] /= runs
        
    return avg_rewards, avg_pulls

# --- Layout ---
st.title("🎰 Multi-Armed Bandit (MAB) Performance Dashboard")
st.markdown("Deep dive into the **Explore-Exploit** dilemma using **Monte Carlo Simulation** to compare classic A/B Testing against advanced Bandit Algorithms.")

with st.spinner("⏳ Running massive parallel simulations in the background..."):
    # Convert numpy array to tuple for Streamlit caching
    methods = ["A/B Test", "Optimistic", "ε-Greedy", "Softmax", "UCB", "Thompson"]
    avg_rewards, avg_pulls = run_mab_simulation(budget, runs, tuple(true_means.tolist()), epsilon, softmax_tau, ucb_c)

total_rewards = {m: np.sum(avg_rewards[m]) for m in methods}
max_mean = np.max(true_means)
optimal_total = budget * max_mean
regrets = {m: optimal_total - total_rewards[m] for m in methods}

# Find extremes
best_method = min(regrets, key=regrets.get)
worst_method = max(regrets, key=regrets.get)

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("✨ Optimal Hindsight Reward", f"${optimal_total:,.0f}", help="Expected reward if the absolute best arm was selected 100% of the time.")
col2.metric(f"🏆 Best Strategy: {best_method}", f"${total_rewards[best_method]:,.0f}", delta=f"Regret: -${regrets[best_method]:,.0f}", delta_color="normal")
col3.metric(f"💔 Worst Strategy: {worst_method}", f"${total_rewards[worst_method]:,.0f}", delta=f"Regret: -${regrets[worst_method]:,.0f}", delta_color="inverse")
col4.metric("📉 Standard A/B Test", f"${total_rewards['A/B Test']:,.0f}", delta=f"Regret: -${regrets['A/B Test']:,.0f}", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# Tabs Container
tab1, tab2, tab3 = st.tabs(["📊 Performance Charts", "📈 Detailed Data Summary", "💡 Algorithm Concepts"])

with tab1:
    st.markdown("### Visual Algorithm Analytics")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 📉 Cumulative Regret (Lower is better)")
        st.markdown("<span style='font-size:0.9em;color:gray;'>A flattening curve indicates the algorithm has successfully locked onto the best arm.</span>", unsafe_allow_html=True)
        regret_df = pd.DataFrame()
        optimal_cum = np.arange(1, budget + 1) * max_mean
        for m in methods:
            regret_df[m] = optimal_cum - np.cumsum(avg_rewards[m])
        st.line_chart(regret_df, height=350, use_container_width=True)
        
    with c2:
        st.markdown("#### 🎰 Arm Pull Allocation")
        st.markdown("<span style='font-size:0.9em;color:gray;'>Superior algorithms allocate the vast majority of resources to the optimal arm.</span>", unsafe_allow_html=True)
        pulls_df = pd.DataFrame(avg_pulls).T
        pulls_df.columns = [f"Bandit A ({mean_A:.2f})", f"Bandit B ({mean_B:.2f})", f"Bandit C ({mean_C:.2f})"]
        st.bar_chart(pulls_df, height=350, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 Final Total Regret Comparison")
    st.markdown("<span style='font-size:0.9em;color:gray;'>A direct macro-comparison of accumulated regrets at the end of the simulation.</span>", unsafe_allow_html=True)
    regret_bar_df = pd.DataFrame({
        "Method": methods,
        "Total Regret ($)": [regrets[m] for m in methods]
    }).set_index("Method")
    st.bar_chart(regret_bar_df, height=250, use_container_width=True)

with tab2:
    st.subheader("📝 Class Comparison Table")
    
    notes = {
        "A/B Test": "Simple but wasteful (Burned budget on inferior arms).",
        "Optimistic": "Front-loaded exploration without complex math.",
        "ε-Greedy": "Easy baseline but permanently leaks regret.",
        "Softmax": "Smooth probabilistic control driven by temperature.",
        "UCB": "Efficient uncertainty-driven exploration bounds.",
        "Thompson": "Bayesian probability theory; the practical gold standard."
    }
    styles = {
        "A/B Test": "Static",
        "Optimistic": "Implicit",
        "ε-Greedy": "Random",
        "Softmax": "Probabilistic",
        "UCB": "Confidence-based",
        "Thompson": "Bayesian"
    }

    comparison_data = []
    for m in methods:
        comparison_data.append({
            "Method": m,
            "Exploration Style": styles[m],
            "Total Expected Reward": f"${total_rewards[m]:,.0f}",
            "Total Regret": f"${regrets[m]:,.0f}",
            "Notes / Key Takeaway": notes[m]
        })

    df = pd.DataFrame(comparison_data).set_index("Method")
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=250)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.subheader("🧠 Understanding Explore-Exploit Strategies")
    
    with st.expander("� A/B Test (Static Testing)"):
         st.write("A highly rigid and conservative approach. It splits a fixed testing budget (e.g., first 20% of pulls) perfectly between a subset of arms. While it guarantees finding a good option eventually, it completely lacks dynamic adaptability and wastes massive budget on clear losers during the test phase.")
         
    with st.expander("🚀 Optimistic Initial Values"):
         st.write("Initializes all Q-values with unrealistically high expectations (e.g., $5.0 out of $1.0 max). When the machines inevitably disappoint the algorithm, it naturally forces exploration by rotating to other 'hopeful' arms. This achieves exploration without relying on random number generation.")
         
    with st.expander("🎲 ε-Greedy"):
         st.write("A hyperparameter epsilon (ε) controls exploration. 90% of the time, the algorithm acts greedily. 10% of the time, it pulls a completely random arm. While effective, its fatal flaw is that it continues to blindly explore even after perfectly identifying the top arm, causing linear lifelong regret.")
         
    with st.expander("�️ Softmax (Boltzmann Distribution)"):
         st.write("Instead of purely uniform random exploration, Softmax assigns sampling probabilities based on the expected values of the arms. A Temperature (Tau) parameter controls the randomness. Better arms get pulled much more, but underperforming arms still retain a non-zero probability of being selected.")
         
    with st.expander("🎯 UCB (Upper Confidence Bound)"):
         st.write("Explores based on both average reward AND uncertainty. If an arm has rarely been pulled, its 'potential' (confidence bound) inflates drastically. This aggressively drives the algorithm to explore unknown options. Once UCB maps the landscape, it locks onto the optimal arm very efficiently.")
         
    with st.expander("📊 Thompson Sampling"):
         st.write("A probabilistic algorithm grounded in Bayes' Theorem. It maintains a distinct Beta Distribution curve for each arm. In practice, this fluidly balances exploration and exploitation, often crushing other metrics in real-world environments like clinical trials and ad targeting.")
