
# Distribution Lab — CLT (and when it fails) + Exponential Memorylessness

An interactive Streamlit app that demonstrates:
- **Central Limit Theorem (CLT):** how the sampling distribution of the mean approaches Normal as sample size `n` increases (Uniform, Exponential, Bernoulli).
- **Cauchy counterexample:** CLT **fails** when the base distribution has no finite mean/variance (Cauchy).
- **Memorylessness (Exponential):** verifies `P(X > s + t | X > s) = P(X > t)` via simulation.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud
1. Create a **public GitHub repo**, add `streamlit_app.py`, `requirements.txt`, and this `README.md`.
2. Go to https://share.streamlit.io/ (Streamlit Community Cloud), connect GitHub, select the repo, and choose `streamlit_app.py` as the entry point.
3. Click **Deploy** — you’ll get a public URL to share.

## What to try
- Switch between **Uniform / Exponential / Bernoulli / Cauchy**.
- Increase **n** and **R** to watch sample means stabilize (or not, for Cauchy).
- On the **Bonus** tab, set distribution to **Exponential** and confirm the **memoryless** property.
