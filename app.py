# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Page config
st.set_page_config(page_title="Stats Explorer", page_icon=":bar_chart:", layout="centered")

st.title("Stats Explorer")
st.caption("Play with a classic statistical idea (CLT), and optionally upload your own data to see CLT and uncertainty on real values.")

tab_clt, tab_data = st.tabs(["CLT Playground", "Your Data (Upload)"])


# TAB 1 - CLT PLAYGROUND (Random distributions, fully self-contained)

with tab_clt:
    st.subheader("Central Limit Theorem Playground")

    st.write(
        "Pick a base distribution, draw many samples, and watch the distribution of the "
        "**sample means** become bell-shaped as the sample size grows."
    )

    colA, colB = st.columns(2)
    with colA:
        dist = st.selectbox("Base distribution", ["Exponential(λ=1)", "Uniform(0,1)", "Poisson(λ=4)"])
        seed = st.number_input("Random seed", value=42, step=1)
    with colB:
        n = st.slider("Sample size per draw (n)", 1, 500, 30)
        draws = st.slider("Number of draws", 100, 5000, 1000)

    rng = np.random.default_rng(int(seed))

    if dist.startswith("Exponential"):
        base = rng.exponential(1.0, size=(draws, n))
        true_mean, true_var = 1.0, 1.0
    elif dist.startswith("Uniform"):
        base = rng.uniform(0.0, 1.0, size=(draws, n))
        true_mean, true_var = 0.5, 1.0 / 12.0
    else:
        lam = 4.0
        base = rng.poisson(lam, size=(draws, n)).astype(float)
        true_mean, true_var = lam, lam

    sample_means = base.mean(axis=1)

    # 1) Base distribution
    st.markdown("**1) Base distribution**")
    fig1, ax1 = plt.subplots()
    ax1.hist(base.flatten(), bins=50)
    ax1.set_xlabel("Value"); ax1.set_ylabel("Frequency")
    st.pyplot(fig1, use_container_width=True)

    # 2) Distribution of sample means
    st.markdown("**2) Distribution of sample means** - becomes more normal as n increases")
    fig2, ax2 = plt.subplots()
    ax2.hist(sample_means, bins=50)
    ax2.set_xlabel("Sample mean"); ax2.set_ylabel("Frequency")
    st.pyplot(fig2, use_container_width=True)

    # 3) Standardized means vs Normal(0,1)
    st.markdown("**3) Standardized sample means vs Normal(0,1)**")
    z = (sample_means - true_mean) / np.sqrt(true_var / n)
    ks_stat, ks_p = stats.kstest(z, 'norm')
    st.caption(f"KS test vs Normal(0,1): statistic = {ks_stat:.3f}, p-value = {ks_p:.3f} "
               "(higher p ≈ closer to normal)")

    fig3, ax3 = plt.subplots()
    ax3.hist(z, bins=50, density=True)
    x = np.linspace(-4, 4, 400)
    ax3.plot(x, stats.norm.pdf(x))
    ax3.set_xlabel("z"); ax3.set_ylabel("Density")
    st.pyplot(fig3, use_container_width=True)


# TAB 2 — YOUR DATA (Upload) - CLT & Uncertainty on your CSV

with tab_data:
    st.subheader("Upload your CSV and explore CLT on a numeric column")
    st.caption("Expected columns for richer demos: 'revenue', 'top', and optionally 'platform' (e.g., mobile/desktop).")

    file = st.file_uploader("Choose a CSV file", type=["csv"])
    if file is None:
        st.write("→ Upload a file to get started.")
    else:
        # Load and preview
        df = pd.read_csv(file)
        st.write("**Preview:**")
        st.dataframe(df.head(), use_container_width=True)

        # Select a numeric column to study
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            st.error("No numeric columns found in the uploaded file.")
            st.stop()

        # Sensible default: revenue if present
        default_col = "revenue" if "revenue" in numeric_cols else numeric_cols[0]
        col_to_study = st.selectbox("Choose a numeric column to study", numeric_cols, index=numeric_cols.index(default_col))

        # Clean selected series
        series = pd.to_numeric(df[col_to_study], errors="coerce").dropna().to_numpy()
        N = len(series)
        if N < 10:
            st.error("Not enough numeric values to analyze. Please upload more data.")
            st.stop()

        st.divider()
        st.markdown("### A) Central Limit Theorem — on your data")

        c1, c2, c3 = st.columns(3)
        with c1:
            n_boot = st.slider("Bootstrap sample size (n)", 5, min(1000, max(10, N)), min(50, N), step=5)
        with c2:
            n_draws = st.slider("Number of bootstrap draws", 200, 10000, 2000, step=200)
        with c3:
            seed_boot = st.number_input("Random seed (bootstrap)", value=123, step=1)

        rng = np.random.default_rng(int(seed_boot))

        # 1) Raw distribution
        st.markdown("**1) Raw distribution**")
        fig_raw, ax_raw = plt.subplots()
        ax_raw.hist(series, bins=50)
        ax_raw.set_xlabel(col_to_study); ax_raw.set_ylabel("Frequency")
        st.pyplot(fig_raw, use_container_width=True)

        # 2) Bootstrap distribution of the sample mean
        means = []
        for _ in range(n_draws):
            idx = rng.integers(0, N, size=n_boot)
            means.append(series[idx].mean())
        means = np.array(means)

        st.markdown("**2) Distribution of bootstrap sample means**")
        fig_boot, ax_boot = plt.subplots()
        ax_boot.hist(means, bins=50, density=True)

        # Overlay Normal(mu_hat, sigma_hat/sqrt(n))
        mu_hat = series.mean()
        sigma_hat = series.std(ddof=1)
        x = np.linspace(means.min(), means.max(), 400)
        ax_boot.plot(x, stats.norm.pdf(x, loc=mu_hat, scale=sigma_hat / np.sqrt(n_boot)))
        ax_boot.set_xlabel("Sample mean"); ax_boot.set_ylabel("Density")
        st.pyplot(fig_boot, use_container_width=True)

        # 95% CI for the mean (bootstrap percentile)
        ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])
        st.caption(
            f"Estimated mean of **{col_to_study}** ≈ {mu_hat:.4f}. "
            f"95% CI (bootstrap): [{ci_lo:.4f}, {ci_hi:.4f}]. "
            "As n increases, the mean's distribution tightens and looks more normal — CLT in action."
        )

        

        st.divider()
        # C) Regression summaries in an expander

        with st.expander("Show regression summaries (simple & with controls)"):
            # simple regression if both columns exist
            if set(["revenue", "top"]).issubset(df.columns):
                dfr = df.dropna(subset=["revenue", "top"]).copy()
                Xs = sm.add_constant(dfr[["top"]])
                ys = dfr["revenue"]
                model_simple = sm.OLS(ys, Xs, missing="drop").fit()
                st.write("**Simple OLS (revenue ~ top)**")
                st.text(model_simple.summary())
            else:
                st.write("Simple OLS skipped (need 'revenue' and 'top').")

            # multiple regression with any of browser/platform/site if present
            cat_cols = [c for c in ["browser", "platform", "site"] if c in df.columns]
            if cat_cols and "revenue" in df.columns and "top" in df.columns:
                dfrm = df.dropna(subset=["revenue", "top"]).copy()
                dfrm = pd.get_dummies(dfrm, columns=cat_cols, drop_first=True)
                X_cols = [c for c in dfrm.columns if c != "revenue"]
                Xm = sm.add_constant(dfrm[X_cols].astype(float))
                ym = dfrm["revenue"].astype(float)
                model_multi = sm.OLS(ym, Xm, missing="drop").fit()
                st.write("**Multiple OLS (with controls)**")
                st.text(model_multi.summary())
            else:
                st.write("Multiple OLS skipped (need at least one of 'browser'/'platform'/'site' plus 'revenue' & 'top').")

