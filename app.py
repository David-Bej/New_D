import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Covariance Analyzer",
                   layout="wide")

st.title("📊 Portfolio Covariance Matrix Analyzer")

st.write("""
Upload a CSV containing historical adjusted closing prices.

Each column should represent one asset.
""")

uploaded = st.file_uploader(
    "Upload Price Data (.csv)",
    type=["csv"]
)

if uploaded:

    prices = pd.read_csv(uploaded,index_col=0)

    st.subheader("Price Data")

    st.dataframe(prices.head())

    returns = prices.pct_change().dropna()

    st.subheader("Daily Returns")

    st.dataframe(returns.head())

    mean_returns = returns.mean()*252

    cov = returns.cov()*252

    corr = returns.corr()

    st.subheader("Annualized Covariance Matrix")

    fig,ax=plt.subplots(figsize=(8,6))

    sns.heatmap(cov,
                annot=True,
                cmap="RdBu_r",
                ax=ax)

    st.pyplot(fig)

    st.subheader("Correlation Matrix")

    fig2,ax2=plt.subplots(figsize=(8,6))

    sns.heatmap(corr,
                annot=True,
                cmap="coolwarm",
                ax=ax2)

    st.pyplot(fig2)

    assets=len(mean_returns)

    simulations=10000

    results=np.zeros((3,simulations))

    weights_record=[]

    for i in range(simulations):

        weights=np.random.random(assets)

        weights/=weights.sum()

        weights_record.append(weights)

        portfolio_return=np.sum(mean_returns*weights)

        portfolio_vol=np.sqrt(
            np.dot(weights.T,
                   np.dot(cov,weights))
        )

        sharpe=portfolio_return/portfolio_vol

        results[0,i]=portfolio_return

        results[1,i]=portfolio_vol

        results[2,i]=sharpe

    portfolios=pd.DataFrame({

        "Return":results[0],
        "Risk":results[1],
        "Sharpe":results[2]

    })

    st.subheader("Monte Carlo Efficient Frontier")

    fig=px.scatter(

        portfolios,

        x="Risk",

        y="Return",

        color="Sharpe",

        color_continuous_scale="Viridis"

    )

    st.plotly_chart(fig,use_container_width=True)

    max_sharpe_idx=portfolios["Sharpe"].idxmax()

    min_risk_idx=portfolios["Risk"].idxmin()

    st.subheader("Optimal Portfolios")

    col1,col2=st.columns(2)

    with col1:

        st.success("Maximum Sharpe Portfolio")

        st.metric("Expected Return",

                  f"{portfolios.loc[max_sharpe_idx,'Return']:.2%}")

        st.metric("Risk",

                  f"{portfolios.loc[max_sharpe_idx,'Risk']:.2%}")

        st.metric("Sharpe",

                  round(portfolios.loc[max_sharpe_idx,'Sharpe'],2))

    with col2:

        st.info("Minimum Variance Portfolio")

        st.metric("Expected Return",

                  f"{portfolios.loc[min_risk_idx,'Return']:.2%}")

        st.metric("Risk",

                  f"{portfolios.loc[min_risk_idx,'Risk']:.2%}")

        st.metric("Sharpe",

                  round(portfolios.loc[min_risk_idx,'Sharpe'],2))

    st.subheader("Portfolio Weights")

    best=pd.DataFrame({

        "Asset":prices.columns,

        "Weight":weights_record[max_sharpe_idx]

    })

    st.dataframe(best)

    fig=px.pie(best,

               values="Weight",

               names="Asset",

               title="Maximum Sharpe Allocation")

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Quantitative Statistics")

    st.write(pd.DataFrame({

        "Annual Return":mean_returns,

        "Annual Volatility":np.sqrt(np.diag(cov))

    }))

    st.subheader("Qualitative Interpretation")

    avg_cov=cov.values[np.triu_indices(len(cov),1)].mean()

    if avg_cov>0.08:

        st.warning("""

High covariance detected.

• Assets tend to move together.

• Diversification benefits are limited.

• Portfolio volatility remains relatively high.

""")

    elif avg_cov>0.03:

        st.info("""

Moderate covariance detected.

• Some diversification exists.

• Risk reduction is moderate.

• Portfolio is reasonably diversified.

""")

    else:

        st.success("""

Low covariance detected.

• Assets behave independently.

• Strong diversification.

• Covariance significantly reduces overall portfolio risk.

""")

    st.subheader("How Covariance Impacts Portfolio Risk")

    vol=[]

    cov_scale=np.linspace(0.2,2.0,25)

    equal=np.repeat(1/assets,assets)

    for s in cov_scale:

        v=np.sqrt(equal.T@(cov*s)@equal)

        vol.append(v)

    fig=go.Figure()

    fig.add_trace(go.Scatter(

        x=cov_scale,

        y=vol,

        mode="lines+markers"

    ))

    fig.update_layout(

        title="Sensitivity of Portfolio Risk to Covariance",

        xaxis_title="Covariance Multiplier",

        yaxis_title="Portfolio Volatility"

    )

    st.plotly_chart(fig,use_container_width=True)

else:

    st.info("Upload a CSV to begin.")
