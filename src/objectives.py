# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:46:19 2025

@author: AriSpiesberger
"""
import numpy as np
import pandas as pd
def sharpe_performance_function(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """Sharpe ratio performance function"""
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Sharpe ratio
    risk_free_rate = 0.02
    return (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0




def monotonicity_performance_function(weights: np.ndarray, returns: pd.DataFrame, 
                                    threshold: float = 0.001) -> float:
    """
    Monotonicity performance function - fraction of days where portfolio return > threshold
    
    Args:
        weights: Portfolio weights
        returns: Daily returns DataFrame
        threshold: Daily threshold (default 0.1% = 0.001)
    
    Returns:
        Fraction of days where portfolio return exceeds threshold
    """
    # Calculate daily portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Count days where return exceeds threshold
    above_threshold = (portfolio_returns > threshold).sum()
    total_days = len(portfolio_returns)
    
    # Return fraction (0 to 1)
    return above_threshold / total_days if total_days > 0 else 0

def sortino_performance_function(weights: np.ndarray, returns: pd.DataFrame, 
                               risk_free_rate: float = 0.02) -> float:
    """
    Sortino ratio performance function - return/downside deviation
    
    Args:
        weights: Portfolio weights
        returns: Daily returns DataFrame
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Sortino ratio
    """
    # Calculate daily portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Annualized portfolio return
    mean_return = portfolio_returns.mean() * 252
    
    # Daily risk-free rate
    daily_rf_rate = risk_free_rate / 252
    
    # Calculate downside deviation (only negative excess returns)
    excess_returns = portfolio_returns - daily_rf_rate
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        # No downside - return a high value
        return 10.0  # Cap at reasonable level
    
    # Downside deviation (annualized)
    downside_variance = (downside_returns ** 2).mean()
    downside_deviation = np.sqrt(downside_variance * 252)
    
    # Sortino ratio
    return (mean_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

def calmar_performance_function(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """
    Calmar ratio performance function - return/maximum drawdown
    
    Args:
        weights: Portfolio weights
        returns: Daily returns DataFrame
    
    Returns:
        Calmar ratio
    """
    # Calculate daily portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Annualized return
    mean_return = portfolio_returns.mean() * 252
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = abs(drawdowns.min())
    
    # Calmar ratio
    return mean_return / max_drawdown if max_drawdown > 0 else 0

def omega_performance_function(weights: np.ndarray, returns: pd.DataFrame, 
                             threshold: float = 0.0) -> float:
    """
    Omega ratio performance function - probability weighted ratio of gains vs losses
    
    Args:
        weights: Portfolio weights
        returns: Daily returns DataFrame
        threshold: Threshold return (default 0%)
    
    Returns:
        Omega ratio
    """
    # Calculate daily portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Separate gains and losses relative to threshold
    excess_returns = portfolio_returns - threshold
    gains = excess_returns[excess_returns > 0]
    losses = excess_returns[excess_returns < 0]
    
    # Sum of gains and absolute sum of losses
    total_gains = gains.sum() if len(gains) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 1e-10  # Avoid division by zero
    
    # Omega ratio
    return total_gains / total_losses

def tail_ratio_performance_function(weights: np.ndarray, returns: pd.DataFrame, 
                                  percentile: float = 0.05) -> float:
    """
    Tail ratio performance function - ratio of average top percentile to average bottom percentile
    
    Args:
        weights: Portfolio weights
        returns: Daily returns DataFrame
        percentile: Percentile for tails (default 5% = 0.05)
    
    Returns:
        Tail ratio (positive tail / negative tail)
    """
    # Calculate daily portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Calculate percentiles
    top_percentile = np.percentile(portfolio_returns, (1 - percentile) * 100)
    bottom_percentile = np.percentile(portfolio_returns, percentile * 100)
    
    # Average returns in each tail
    top_tail_returns = portfolio_returns[portfolio_returns >= top_percentile]
    bottom_tail_returns = portfolio_returns[portfolio_returns <= bottom_percentile]
    
    avg_top_tail = top_tail_returns.mean() if len(top_tail_returns) > 0 else 0
    avg_bottom_tail = bottom_tail_returns.mean() if len(bottom_tail_returns) > 0 else -1e-10
    
    # Tail ratio (handle negative bottom tail)
    return abs(avg_top_tail / avg_bottom_tail) if avg_bottom_tail != 0 else 0

def ulcer_performance_function(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """
    Ulcer Performance Index - return/ulcer index (RMS of drawdowns)
    
    Args:
        weights: Portfolio weights
        returns: Daily returns DataFrame
    
    Returns:
        Ulcer Performance Index
    """
    # Calculate daily portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Annualized return
    mean_return = portfolio_returns.mean() * 252
    
    # Calculate drawdowns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - running_max) / running_max * 100  # In percentage
    
    # Ulcer Index (RMS of drawdowns)
    ulcer_index = np.sqrt((drawdowns ** 2).mean())
    
    # Ulcer Performance Index
    return mean_return / ulcer_index if ulcer_index > 0 else 0

def information_ratio_performance_function(weights: np.ndarray, returns: pd.DataFrame, 
                                         benchmark_weights: np.ndarray = None) -> float:
    """
    Information ratio performance function - active return/tracking error
    
    Args:
        weights: Portfolio weights
        returns: Daily returns DataFrame
        benchmark_weights: Benchmark weights (default: equal weight)
    
    Returns:
        Information ratio
    """
    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Default benchmark: equal weight
    if benchmark_weights is None:
        benchmark_weights = np.ones(len(weights)) / len(weights)
    
    # Calculate benchmark returns
    benchmark_returns = returns.dot(benchmark_weights)
    
    # Active returns
    active_returns = portfolio_returns - benchmark_returns
    
    # Information ratio
    mean_active_return = active_returns.mean() * 252  # Annualized
    tracking_error = active_returns.std() * np.sqrt(252)  # Annualized
    
    return mean_active_return / tracking_error if tracking_error > 0 else 0

def comprehensive_performance_function(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """
    Comprehensive performance function combining multiple metrics
    
    Args:
        weights: Portfolio weights
        returns: Daily returns DataFrame
    
    Returns:
        Weighted combination of performance metrics
    """
    # Calculate individual metrics
    sharpe = sharpe_performance_function(weights, returns)
    sortino = sortino_performance_function(weights, returns)
    monotonicity = monotonicity_performance_function(weights, returns)
    calmar = calmar_performance_function(weights, returns)
    
    # Normalize metrics to prevent any single metric from dominating
    sharpe_norm = np.tanh(sharpe / 2.0)  # Normalize around 2.0
    sortino_norm = np.tanh(sortino / 3.0)  # Normalize around 3.0
    monotonicity_norm = monotonicity  # Already 0-1
    calmar_norm = np.tanh(calmar / 1.0)  # Normalize around 1.0
    
    # Weighted combination
    weights_metrics = [0.3, 0.3, 0.2, 0.2]  # Sharpe, Sortino, Monotonicity, Calmar
    
    return (weights_metrics[0] * sharpe_norm + 
            weights_metrics[1] * sortino_norm + 
            weights_metrics[2] * monotonicity_norm + 
            weights_metrics[3] * calmar_norm)

# Dictionary of all performance functions for easy testing
PERFORMANCE_FUNCTIONS = {
    'sharpe': sharpe_performance_function,
    'monotonicity': monotonicity_performance_function,
    'sortino': sortino_performance_function,
    'calmar': calmar_performance_function,
    'omega': omega_performance_function,
    'tail_ratio': tail_ratio_performance_function,
    'ulcer': ulcer_performance_function,
    'information_ratio': information_ratio_performance_function,
    'comprehensive': comprehensive_performance_function
}

def test_all_performance_functions(returns: pd.DataFrame, weights: np.ndarray = None):
    """
    Test all performance functions with given returns data
    
    Args:
        returns: Daily returns DataFrame
        weights: Portfolio weights (default: equal weight)
    
    Returns:
        Dictionary of performance scores
    """
    if weights is None:
        weights = np.ones(len(returns.columns)) / len(returns.columns)
    
    results = {}
    
    for name, func in PERFORMANCE_FUNCTIONS.items():
        try:
            score = func(weights, returns)
            results[name] = score
            print(f"{name.capitalize():15}: {score:.4f}")
        except Exception as e:
            print(f"{name.capitalize():15}: Error - {e}")
            results[name] = np.nan
    
    return results