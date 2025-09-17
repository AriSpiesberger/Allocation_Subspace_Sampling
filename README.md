# Allocation_Subspace_Sampling
Portfolio Subspace Allocation Expirements

The following expirements are decided to demonstrate the efficacy of sampling in the portfolio optimization from the transformation induced by a correlation based hierarchacal decision tree. 9/17/2025

## Motivation


"As a fund manager, you have some function that maps your investments to predictive results. Is your function good, and secondly, how do you select your inputs?"

Clearly, having an excellent predictive function is crucial towards generating any alpha (or realizing some features of the stochastic process you are interested in). We simply argue here that the second problem is also clearly valuable, and provide some useful solutions. 

An Example: 
Let's set up a simple toy example. Suppose your objective function is the traditional Sharpe ratio, and you can either buy or not buy each asset (with equal weighting). Now imagine you had a predictive model that could perfectly forecast the Sharpe ratio of any portfolio. You might think you'd be in great shape. But, as we'll see: that's not the whole story.

The sharpe ratio for any asset (i) is defined as

$$S_i = \frac{\mathbb{E}[R_i] - r_f}{\sigma_i}$$

And for a process 

$$S_p = \frac{\mathbb{E}[R_p] - r_f}{\sigma_p} = \frac{\mathbf{w}^T \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}}$$

To those who dont have a math or statistics background, what is happening here is that ...

Clearly the covariance in the denominator shows that Sharpe is not additive. In fact, there is no guarentee of linearity at all since you can't decompose portfolio Sharpe into simple functions of individual asset Sharpes. Standard properties we might expect from a metric space, like the triangle inequality, don't hold. This means you can have two assets with mediocre individual Sharpe ratios that, when combined, produce a portfolio with a much higher Sharpe than either asset alone. The whole is genuinely different from the sum of its parts. 


Here is a concrete example


| Asset   | Return | Std  | Sharpe |
|---------|--------|------|--------|
| Asset 1 | 0.08   | 0.10 | 0.80   |
| Asset 2 | 0.14   | 0.20 | 0.70   |
| Asset 3 | 0.06   | 0.10 | 0.60   |
| Asset 4 | 0.05   | 0.10 | 0.50   |

Suppose all assets are uncorrelated except for Assets 3 and 4, which have a strong negative covariance of -0.8. 

Thus, since Sharpe ratios do not induce a metric space and correlations introduce nontrivial interactions, sampling is tricky and expensive to search over all possible asset combinations would generally be required to find the optimal portfolio. 

In this toy example, the optimal portfolio is to buy only Assets 3 and 4, resulting in a portfolio Sharpe ratio of 1.72: far higher than any single asset achieves individually.


This simple example illustrates three key points:

1) Even perfect knowledge of individual asset Sharpe ratios does not identify the optimal portfolio.
2) Optimal combinations often rely on hidden structure (such as specific correlations) that are invisible when examining assets in isolation.
3) As the number of assets grows, optimal portfolios can become increasingly rare and sparse, making naive search methods impractical.

#### Computational complexity problem
The evidence above suggests a compelling need for more complete searches over the input space. Although sharpe was our selected toy example, there are many situations where one would want to search the space in some sense of completion. Furthermore, in our case we have $2^4$ portfolio compositions, however there may be as many as $$2^n$$ compositions for an unconstrained portfolio of n assets, (even in this binary case). In realistic portfolio optimization, you would be optimizing in n dimensions a continuous space. The reader should be reminded that the amount of atoms in the universe is $$2^{84}$$ or so, so we may easily be able to reach a place where even in parallel a computer cannot simply brute force all calculations. 

We thus claim that under black-box conditions of our function f, with little assumptions, we must create a sampling method that will be able to reveal optimal, or near optimal portfolios, as well as informing our understanding of the distribution around them. 