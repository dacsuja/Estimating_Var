# Estimating Value at Risk (VaR) with Copula GARCH

This repository provides a Python implementation for estimating the **Value at Risk (VaR)** of a financial portfolio using a **Copula GARCH** approach. 

Standard VaR models often assume normal distributions and fail to capture the complex dependencies and volatility clustering present in real-world financial markets. By combining GARCH (Generalized Autoregressive Conditional Heteroskedasticity) to model the time-varying volatility of individual assets, and Copulas to model the dependence structure between them, this approach provides a much more robust risk estimation—particularly during market downturns.

## 📁 Repository Structure

* `Estimating VaR with Copula Garch.py`: The core Python script containing the data processing, model fitting, and VaR estimation logic.
* `figure1.png`: Visualization output (e.g., Asset returns / Volatility clustering).
* `figure2.png`: Visualization output (e.g., Copula dependence structure / scatter plots).
* `figure3.png`: Visualization output (e.g., Estimated VaR thresholds vs. actual portfolio returns).

## 🚀 Features

* **Volatility Modeling:** Fits GARCH models to individual asset returns to capture volatility clustering and heavy tails.
* **Dependence Modeling:** Utilizes Copulas to model the joint multivariate distribution of the assets without relying on the assumption of normality.
* **Risk Estimation:** Calculates the Value at Risk (VaR) at specified confidence intervals.
* **Visualization:** Automatically generates plots to visualize returns, volatility, and VaR breaches.

## 🛠️ Prerequisites

To run the script, you will need Python 3.x and the following libraries installed. You can install the standard quantitative finance stack via `pip`:

```bash
pip install numpy pandas matplotlib scipy arch