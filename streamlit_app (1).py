
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Distribution Lab: CLT & More", layout="wide")
st.title("📈 Distribution Lab — Central Limit Theorem (and when it fails)")
st.caption("Interactively explore sampling distributions, see the CLT in action, and test the exponential memoryless property.")

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.header("1) Choose base distribution")
dist_name = st.sidebar.selectbox("Distribution", ["Uniform", "Exponential", "Bernoulli", "Cauchy"], index=0)

if dist_name == "Uniform":
    a = st.sidebar.number_input("a (min)", value=0.0, step=0.1)
    b = st.sidebar.number_input("b (max)", value=1.0, step=0.1)
    if b <= a:
        st.sidebar.error("Require b > a")
elif dist_name == "Exponential":
    lam = st.sidebar.number_input("rate λ (>0)", value=1.0, min_value=0.0001, step=0.1)
elif dist_name == "Bernoulli":
    p = st.sidebar.slider("p (0..1)", 0.0, 1.0, 0.5, 0.01)
elif dist_name == "Cauchy":
    loc = st.sidebar.number_input("location", value=0.0, step=0.1)
    scale = st.sidebar.number_input("scale (>0)", value=1.0, min_value=0.0001, step=0.1)

st.sidebar.header("2) Sampling controls")
n = st.sidebar.slider("Sample size per replication (n)", 1, 1000, 30, 1)
R = st.sidebar.slider("Number of replications (R)", 100, 20000, 5000, 100)
seed = st.sidebar.number_input("Random seed (optional)", value=0, step=1)
if seed != 0:
    np.random.seed(int(seed))

# --------------------------
# Helper: draw samples
# --------------------------
def draw_rv(dist, n, R):
    if dist == "Uniform":
        X = np.random.uniform(a, b, size=(R, n))
        mu, sigma = (a + b) / 2.0, (b - a) / np.sqrt(12.0)
    elif dist == "Exponential":
        X = np.random.exponential(1.0 / lam, size=(R, n))
        mu, sigma = 1.0 / lam, 1.0 / lam
    elif dist == "Bernoulli":
        X = np.random.binomial(1, p, size=(R, n)).astype(float)
        mu, sigma = p, np.sqrt(p * (1 - p))
    elif dist == "Cauchy":
        X = stats.cauchy.rvs(loc=loc, scale=scale, size=(R, n))
        mu, sigma = np.nan, np.nan  # undefined/doesn't exist
    return X, mu, sigma

X, mu, sigma = draw_rv(dist_name, n, R)
means = X.mean(axis=1)

# --------------------------
# Layout
# --------------------------
tab1, tab2 = st.tabs(["Sampling Distribution of the Mean (CLT)", "Bonus: Exponential Memorylessness"])

with tab1:
    colL, colR = st.columns([1, 1])

    # Left: show base distribution (single big sample)
    with colL:
        st.subheader("Base distribution (raw draws)")
        raw = X.ravel() if X.size <= 200000 else X.ravel()[:200000]
        fig1, ax1 = plt.subplots(figsize=(6,4))
        ax1.hist(raw, bins=40, density=True, alpha=0.6)
        ax1.set_title(f"{dist_name} — raw draws (subset)")
        ax1.set_xlabel("x"); ax1.set_ylabel("density")
        st.pyplot(fig1)

        # Theoretical notes
        if dist_name != "Cauchy":
            st.markdown(f"**Theoretical mean (μ):** {mu:.4f}  •  **Theoretical sd (σ):** {sigma:.4f}")
        else:
            st.markdown("**Theoretical mean/sd:** *do not exist* for the Cauchy distribution.")

    # Right: sampling distribution of the sample mean
    with colR:
        st.subheader("Sampling distribution of the mean (over R replications)")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.hist(means, bins=40, density=True, alpha=0.65, label="sample means")

        # Overlay normal approximation if CLT assumptions OK
        if dist_name != "Cauchy":
            grid = np.linspace(np.min(means), np.max(means), 400)
            approx = stats.norm.pdf(grid, loc=mu, scale=sigma/np.sqrt(n))
            ax2.plot(grid, approx, linewidth=2.5, label="Normal approx (μ, σ/√n)")
        else:
            ax2.text(0.02, 0.95, "CLT does not apply (no finite mean/variance)", transform=ax2.transAxes, va="top", fontsize=10)

        ax2.set_title(f"Distribution of sample means (n={n}, R={R})")
        ax2.set_xlabel("sample mean"); ax2.set_ylabel("density")
        ax2.legend()
        st.pyplot(fig2)

        # Normality check (Q–Q)
        st.markdown("**Q–Q plot vs Normal** (for sample means)")
        figqq, axqq = plt.subplots(figsize=(6,4))
        if dist_name != "Cauchy":
            # use theoretical normal with (μ, σ/√n)
            (theo, samp), (slope, intercept, r_q) = stats.probplot(
                means, dist="norm", sparams=(mu, sigma/np.sqrt(n))
            )
        else:
            # fall back to empirical normalization just to visualize
            m_emp, s_emp = np.mean(means), np.std(means)
            (theo, samp), (slope, intercept, r_q) = stats.probplot(
                (means - m_emp)/s_emp, dist="norm"
            )
        axqq.scatter(theo, samp, s=10, alpha=0.6)
        minv, maxv = np.min(theo), np.max(theo)
        axqq.plot([minv, maxv], [minv, maxv], linestyle="--")
        axqq.set_title("Q–Q plot (sample means vs Normal)")
        axqq.set_xlabel("Theoretical quantiles"); axqq.set_ylabel("Ordered sample means")
        st.pyplot(figqq)

        # Stats box
        st.markdown("**Summary (sample means):**")
        st.write({
            "empirical mean of means": float(np.mean(means)),
            "empirical sd of means": float(np.std(means)),
            "theoretical sd (σ/√n)": None if np.isnan(sigma) else float(sigma/np.sqrt(n)),
        })

    st.info("**Central Limit Theorem (CLT):** For many base distributions with finite mean/variance (e.g., Uniform, Exponential, Bernoulli), the distribution of sample means tends toward Normal as n increases. "
            "For heavy‑tailed distributions like **Cauchy**, the CLT assumptions fail—sample means do **not** settle down.")

with tab2:
    st.subheader("Exponential distribution is memoryless")
    st.caption("For Exponential(λ), P(X > s + t | X > s) = P(X > t). This is special to the exponential distribution.")

    ss = st.slider("s (elapsed)", 0.0, 5.0, 1.0, 0.1)
    tt = st.slider("t (additional)", 0.0, 5.0, 1.0, 0.1)
    Nsim = st.slider("Simulations", 1000, 200000, 50000, 1000)

    if dist_name != "Exponential":
        st.warning("Set the base distribution to **Exponential** in the sidebar for a true memoryless property.")
    # simulate from exponential with current λ or default λ=1
    lam_use = lam if dist_name == "Exponential" else 1.0
    Xs = np.random.exponential(1.0/lam_use, size=Nsim)
    A = Xs > ss
    lhs = np.mean(Xs[A] > ss + tt) if A.any() else np.nan
    rhs = np.mean(Xs > tt)
    st.write({
        "Estimated P(X > s+t | X > s)": float(lhs),
        "Estimated P(X > t)": float(rhs),
        "Rate λ": float(lam_use)
    })
    figm, axm = plt.subplots(figsize=(6,3.8))
    axm.bar(["P(X>s+t | X>s)", "P(X>t)"], [lhs, rhs])
    axm.set_ylim(0,1)
    axm.set_title("Memorylessness check (they should match)")
    st.pyplot(figm)

st.markdown("---")
st.caption("Tips: Increase n to see the CLT kick in for Uniform/Exponential/Bernoulli. Switch to Cauchy to see the CLT fail (sample means remain heavy‑tailed).")
