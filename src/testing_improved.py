import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
import time  # MODIFIED: Import the time module
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics"""
    weights: np.ndarray
    performance_score: float
    iteration: int

class AssetDataGenerator:
    """Handles both real and simulated asset data generation"""
    
    @staticmethod
    def get_sp500_assets(n_assets: int, period: str = "2y") -> Tuple[pd.DataFrame, List[str]]:
        """Download real S&P 500 asset data"""
        # Sample of S&P 500 tickers for testing
        sp500_sample = [
            # S&P 500 & Major US Stocks
            'MSFT', 'NVDA', 'AAPL', 'AMZN', 'META', 'AVGO', 'GOOGL', 'TSLA', 'GOOG', 'JPM', 
            'V', 'LLY', 'NFLX', 'MA', 'COST', 'XOM', 'WMT', 'PG', 'JNJ', 'HD', 'ABBV', 
            'BAC', 'UNH', 'KO', 'PM', 'CRM', 'ORCL', 'CSCO', 'GE', 'PLTR', 'IBM', 'WFC', 
            'ABT', 'MCD', 'CVX', 'LIN', 'NOW', 'DIS', 'ACN', 'T', 'ISRG', 'MRK', 'UBER', 
            'GS', 'INTU', 'VZ', 'AMD', 'ADBE', 'RTX', 'PEP', 'BKNG', 'TXN', 'QCOM', 'PGR', 
            'CAT', 'SPGI', 'AXP', 'MS', 'BSX', 'BA', 'TMO', 'TJX', 'NEE', 'AMGN', 'HON', 
            'BLK', 'C', 'UNP', 'GILD', 'CMCSA', 'AMAT', 'ADP', 'PFE', 'SYK', 'DE', 'LOW', 
            'ETN', 'PANW', 'DHR', 'COF', 'TMUS', 'MMC', 'VRTX', 'COP', 'ADI', 'MDT', 'CB', 
            'CRWD', 'MU', 'LRCX', 'APH', 'KLAC', 'CME', 'MO', 'BX', 'ICE', 'AMT', 'LMT', 
            'SO', 'PLD', 'ANET', 'BMY', 'TT', 'SBUX', 'ELV', 'DUK', 'WELL', 'MCK', 'CEG', 
            'INTC', 'CDNS', 'CI', 'AJG', 'WM', 'PH', 'MDLZ', 'EQIX', 'SHW', 'MMM', 'KKR', 
            'TDG', 'ORLY', 'CVS', 'SNPS', 'AON', 'CTAS', 'CL', 'MCO', 'ZTS', 'MSI', 'PYPL', 
            'NKE', 'WMB', 'GD', 'UPS', 'DASH', 'CMG', 'HCA', 'PNC', 'USB', 'HWM', 'ECL', 
            'EMR', 'ITW', 'FTNT', 'AZO', 'NOC', 'JCI', 'BK', 'REGN', 'ADSK', 'EOG', 'TRV', 
            'ROP', 'APD', 'NEM', 'MAR', 'HLT', 'RCL', 'CSX', 'APO', 'CARR', 'WDAY', 'ABNB', 
            'AEP', 'COIN', 'FCX', 'TRGP', 'PSX', 'KMI', 'MPC', 'TPL', 'BKR', 'CTRA', 'OKE',
            'EXC', 'AIG', 'MET', 'PRU', 'ALL', 'SCHW', 'FDX', 'GM', 'F', 'TGT', 'KHC', 
            'KMB', 'DAL', 'UAL', 'LUV', 'AAL', 'MAR', 'HLT', 'WYNN', 'LVS', 'EXPE', 
            'CCL', 'NCLH', 'EBAY', 'ETSY', 'ZM', 'DOCU', 'PTON', 'RBLX', 'U', 
            'SHOP', 'SNOW', 'AI', 'DDOG', 'MDB', 'OKTA', 'TWLO', 'NET', 'TEAM', 'ZS',
            'S', 'ROKU', 'SPOT', 'PINS', 'SNAP',
            
            # US Equity Sector ETFs
            'XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC',

            # US Broad Market & Style ETFs
            'SPY', 'IVV', 'VOO', 'VTI', 'QQQ', 'DIA', 'IWM', 'IWB', 'IWD', 'IWF', 'IWN',
            'IWO', 'MDY', 'SCHD', 'VUG', 'VTV', 'VO', 'VB', 'SCHA', 'SCHG', 'SCHX', 'FNDX',

            # International Equity ETFs
            'VEA', 'VWO', 'EFA', 'EEM', 'ACWI', 'VXUS', 'IEFA', 'IEMG', 'IXUS', 'IDEV',
            'EWW', 'EWJ', 'EWZ', 'INDA', 'EWG', 'EWU', 'EWC', 'MCHI', 'KWEB', 'FXI', 
            'RSX', 'VPL', 'VGK', 'VNQ', 'VIGI', 'DEM', 'EDC', 'EZA', 'TUR', 'GREK',

            # Bond & Fixed Income ETFs
            'AGG', 'BND', 'TLT', 'IEF', 'SHY', 'GOVT', 'LQD', 'VCSH', 'VCIT', 'HYG', 
            'JNK', 'BKLN', 'MUB', 'VTEB', 'BNDX', 'EMB', 'VWOB', 'IGOV', 'TIP', 'STIP',
            'SHV', 'BIL', 'BSV', 'BIV', 'BLV', 'MBB', 'VMBS', 'PIM', 'BOND', 'TOTL',

            # Commodity ETFs
            'GLD', 'IAU', 'SLV', 'PPLT', 'PALL', 'USO', 'UCO', 'BNO', 'UNG', 'CORN',
            'WEAT', 'SOYB', 'CANE', 'DBA', 'DBC', 'PDBC', 'GSG', 'GCC', 'USCI', 'CPER',

            # Real Estate (REIT) ETFs
            'VNQ', 'IYR', 'SCHH', 'RWR', 'XLRE', 'REZ', 'REM', 'MORT',

            # Currency ETFs
            'UUP', 'UDN', 'FXY', 'FXE', 'FXB', 'FXC', 'FXA', 'CYB',

            # Alternative & Thematic ETFs
            'ARKK', 'ARKG', 'ARKW', 'ARKQ', 'ARKF', 'SMH', 'SOXX', 'IGV', 'IBB', 'IYT',
            'ITA', 'KBE', 'KRE', 'TAN', 'ICLN', 'PBW', 'BOTZ', 'AIQ', 'LIT', 'URA', 'MJ',
            'IPO', 'BUZZ', 'BETZ', 'ESPO',
            
            # More Large/Mid Cap Stocks from S&P 500/400
            'AMP', 'APA', 'AKAM', 'ALB', 'ARE', 'ATO', 'AVY', 'AWK', 'BALL', 'BBWI', 'BBY',
            'BEN', 'BIIB', 'BIO', 'BR', 'BRO', 'CDW', 'CF', 'CFG', 'CHTR', 'CNC', 'CNP', 
            'CPB', 'CPRT', 'CRL', 'CTSH', 'CTVA', 'DG', 'DGX', 'DHI', 'DPZ', 'DRI',
            'DTE', 'DVN', 'DXCM', 'EA',  'EFX', 'EG', 'EL', 'ENPH', 'EPAM', 'EQR',
            'ESS', 'EVRG', 'EW', 'FAST', 'FFIV', 'FIS', 'FITB',  'FRT', 'FSLR', 'GPN',
            'GRMN', 'GWW', 'HAS', 'HBAN',  'HIG', 'HPE', 'HPQ', 'HRB', 'HST', 'HSY',
            'HUM', 'ILMN', 'INCY', 'IP', 'IPG', 'IQV', 'IRM', 'IVZ', 'JBHT', 'JKHY', 'JPM',
            'KEY', 'KIM', 'KMX', 'KR', 'LEN', 'LH', 'LHX', 'LKQ', 'LYB', 'LYV', 'MAS',
            'MAA', 'MTB', 'MTCH', 'MKC', 'MKTX', 'MLM', 'MOH', 'MOS', 'MPWR', 'NDAQ', 'NI',
            'NTRS', 'NVR', 'NWL', 'NWS', 'NWSA', 'O', 'ODFL', 'OGN', 'OMC', 'ON', 'PAYC',
            'PAYX', 'PCAR', 'PCG', 'PEG', 'PPL', 'PPG', 'PVH', 'QRVO', 'REG', 'RF', 'RL',
            'ROL', 'ROST', 'SBAC', 'SJM', 'SNA', 'STX', 'STLD', 'STE', 'SWK', 'SWKS',
 'SYF', 'TAP', 'TER', 'TFC', 'TFX', 'TSCO', 'TSN', 'TXT', 'TYL', 'ULTA', 'VFC',
 'VICI', 'VRSK', 'VRSN', 'VTR', 'WAB', 'WBD', 'WDC', 'WEC', 'WHR', 'WST', 'WY', 
 'XEL', 'XRAY', 'YUM', 'ZBH']
                
        # Select random subset
        selected_tickers = random.sample(sp500_sample, min(n_assets, len(sp500_sample)))
        
        try:
            # Download data
            print(f"Downloading data for {len(selected_tickers)} assets...")
            data = yf.download(selected_tickers, period=period, progress=False)
            
            # Handle different data structures
            if isinstance(data.columns, pd.MultiIndex):
                if 'Adj Close' in data.columns.get_level_values(0):
                    adj_close_data = data['Adj Close']
                else:
                    adj_close_data = data['Close']
            else:
                adj_close_data = data
            
            # Handle single asset case
            if isinstance(adj_close_data, pd.Series):
                adj_close_data = adj_close_data.to_frame()
                adj_close_data.columns = selected_tickers
            
            # Clean data and calculate returns
            adj_close_data = adj_close_data.dropna()
            
            if adj_close_data.empty:
                raise ValueError("No valid data after cleaning")
            
            returns = adj_close_data.pct_change().dropna()
            successful_tickers = list(returns.columns)
            
            print(f"Successfully processed {len(successful_tickers)} assets")
            return returns, successful_tickers, adj_close_data
            
        except Exception as e:
            print(f"Error downloading real data: {e}. Using simulated data...")
            return AssetDataGenerator.generate_simulated_assets(n_assets)
    
    @staticmethod
    def generate_simulated_assets(n_assets: int, n_periods: int = 500) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
        """Generate simulated asset returns with realistic correlations"""
        print(f"Generating simulated data for {n_assets} assets...")
        
        # Create proper date index
        start_date = pd.Timestamp('2022-01-01')
        date_index = pd.date_range(start=start_date, periods=n_periods, freq='D')
        
        # Generate random correlation matrix
        A = np.random.randn(n_assets, n_assets)
        corr_matrix = np.dot(A, A.T)
        
        # Normalize to correlation matrix
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)
        
        # Generate base returns
        base_returns = np.random.multivariate_normal(
            mean=np.random.uniform(0.0005, 0.002, n_assets),  # Daily returns 0.05% to 0.2%
            cov=corr_matrix * 0.0001,  # Scale to reasonable volatility
            size=n_periods
        )
        
        # Create asset names
        asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
        
        # Create DataFrame with proper date index
        returns_df = pd.DataFrame(base_returns, columns=asset_names, index=date_index)
        
        # Generate price series from returns
        initial_prices = np.random.uniform(50, 200, n_assets)  # Random starting prices
        price_series = pd.DataFrame(index=date_index, columns=asset_names, dtype=float)
        
        for i, asset in enumerate(asset_names):
            price_series.iloc[0, i] = initial_prices[i]
            for j in range(1, len(returns_df)):
                price_series.iloc[j, i] = price_series.iloc[j-1, i] * (1 + returns_df.iloc[j, i])
        
        return returns_df, asset_names, price_series

# Performance functions
def sharpe_performance_function(weights, returns_data):
    """
    Calculates the Sharpe Ratio for a given portfolio.
    Returns -np.inf if the calculation fails.
    """
    try:
        # Your existing calculation logic
        portfolio_return = np.sum(returns_data.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns_data.cov() * 252, weights)))

        # Check for division by zero
        if portfolio_std == 0:
            return -np.inf # Penalize portfolios with zero risk/return

        sharpe_ratio = portfolio_return / portfolio_std
        return sharpe_ratio

    except Exception as e:
        # If any other error occurs during calculation, return a very bad score
        print(f"Warning: Could not calculate Sharpe Ratio. Error: {e}")
        return -np.inf

# Sampling methods
class SamplingMethod(ABC):
    """Abstract base class for sampling methods"""
    
    @abstractmethod
    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        """Generate a portfolio weight sample"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return method name"""
        pass

class RandomUniformSampling(SamplingMethod):
    """Simple random uniform sampling"""
    
    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        n_assets = len(returns.columns)
        weights = np.random.uniform(0, 1, n_assets)
        return weights / weights.sum()
    
    def get_name(self) -> str:
        return "Random Uniform"

class FastHierarchicalSampling(SamplingMethod):
    """Fast hierarchical sampling with equal weight bias"""
    
    def __init__(self, equal_weight_bias: float = 0.1):
        self.equal_weight_bias = equal_weight_bias
        
    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        n_assets = len(returns.columns)
        
        # Start with equal weights
        equal_weights = np.ones(n_assets) / n_assets
        
        # Add some randomness
        random_weights = np.random.uniform(0, 1, n_assets)
        random_weights = random_weights / random_weights.sum()
        
        # Combine with bias towards equal weights
        weights = (self.equal_weight_bias * equal_weights + 
                  (1 - self.equal_weight_bias) * random_weights)
        
        return weights / weights.sum()
    
    def get_name(self) -> str:
        return f"Fast Hierarchical (bias={self.equal_weight_bias})"

class HierarchicalBayesianSampling(SamplingMethod):
    """Hierarchical Bayesian sampling with acquisition function"""
    
    def __init__(self, equal_weight_bias: float = 0.2, acquisition_function: str = 'pi'):
        self.equal_weight_bias = equal_weight_bias
        self.acquisition_function = acquisition_function
        self.history = []
        
    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        n_assets = len(returns.columns)
        
        if iteration < 10 or len(self.history) < 5:
            # Initial random exploration
            weights = np.random.dirichlet(np.ones(n_assets))
        else:
            # Use history to guide sampling
            # Simple Bayesian-inspired approach
            past_weights = np.array([h['weights'] for h in self.history[-20:]])  # Last 20
            past_scores = np.array([h['score'] for h in self.history[-20:]])
            
            # Weight by performance (simple approach)
            if len(past_scores) > 0:
                normalized_scores = (past_scores - past_scores.min() + 1e-8)
                normalized_scores = normalized_scores / normalized_scores.sum()
                
                # Weighted average of good portfolios
                weighted_mean = np.average(past_weights, weights=normalized_scores, axis=0)
                
                # Add exploration noise
                noise = np.random.normal(0, 0.1, n_assets)
                weights = weighted_mean + noise
                weights = np.abs(weights)  # Ensure non-negative
            else:
                weights = np.random.dirichlet(np.ones(n_assets))
        
        # Normalize
        weights = weights / weights.sum()
        
        # Store for next iteration (will be updated externally)
        return weights
    
    def update_history(self, weights: np.ndarray, score: float):
        """Update history with latest result"""
        self.history.append({'weights': weights.copy(), 'score': score})
        
    def get_name(self) -> str:
        return f"Hierarchical Bayesian (bias={self.equal_weight_bias}, acq={self.acquisition_function})"

class PortfolioPredictor:
    """Handles forward performance prediction for portfolios"""
    
    def __init__(self, returns: pd.DataFrame, performance_function: Callable[[np.ndarray, pd.DataFrame], float]):
        self.returns = returns
        self.performance_function = performance_function
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252     # Annualized
        
    def predict_performance(self, weights: np.ndarray) -> float:
        """Predict forward performance for given portfolio weights"""
        return self.performance_function(weights, self.returns)
    
    def get_portfolio_stats(self, weights: np.ndarray) -> Dict:
        """Get detailed portfolio statistics"""
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'performance_score': self.predict_performance(weights)
        }

class TestEnvironment:
    """Main test environment for portfolio optimization experiments"""
    
    def __init__(self, 
                 n_assets: int,
                 performance_function: Callable[[np.ndarray, pd.DataFrame], float],
                 use_real_data: bool = True,
                 # MODIFIED: Add optional parameters to accept pre-loaded data
                 returns_data: Optional[pd.DataFrame] = None,
                 asset_names: Optional[List[str]] = None,
                 price_data: Optional[pd.DataFrame] = None):
        
        self.n_assets = n_assets
        self.use_real_data = use_real_data
        self.performance_function = performance_function
        
        # MODIFIED: Use pre-loaded data if provided, otherwise fetch it
        if returns_data is not None and asset_names is not None and price_data is not None:
            print("Using pre-loaded asset data for this environment.")
            self.returns, self.asset_names, self.price_data = returns_data, asset_names, price_data
        elif use_real_data:
            self.returns, self.asset_names, self.price_data = AssetDataGenerator.get_sp500_assets(n_assets)
        else:
            self.returns, self.asset_names, self.price_data = AssetDataGenerator.generate_simulated_assets(n_assets)
        
        self.predictor = PortfolioPredictor(self.returns, performance_function)
        self.results = {}
        self.optimal_weights = None
        self.optimal_score = None
        
        print(f"Initialized test environment with {len(self.asset_names)} assets")
        print(f"Data period: {len(self.returns)} observations")
        
    def calculate_optimal_portfolio(self):
        """Calculate theoretical optimal portfolio using scipy optimization"""
        try:
            from scipy.optimize import minimize
            
            mean_returns = self.returns.mean() * 252
            cov_matrix = self.returns.cov() * 252
            n_assets = len(mean_returns)
            
            def negative_sharpe(weights):
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                if portfolio_volatility == 0:
                    return 1e10  # Large penalty
                
                risk_free_rate = 0.02
                sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
                return -sharpe
            
            # Constraints and bounds
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_guess = np.ones(n_assets) / n_assets
            
            # Multiple starting points to avoid local optima
            best_result = None
            best_score = 1e10
            
            for _ in range(10):  # Try 10 different starting points
                start_weights = np.random.dirichlet(np.ones(n_assets))
                
                result = minimize(
                    negative_sharpe,
                    start_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-12}
                )
                
                if result.success and result.fun < best_score:
                    best_result = result
                    best_score = result.fun
            
            if best_result and best_result.success:
                self.optimal_weights = best_result.x
                self.optimal_score = -best_result.fun
                
                print(f"\nOptimal Portfolio Found:")
                print(f"Theoretical Maximum Sharpe Ratio: {self.optimal_score:.4f}")
                print("Optimal weights (>1%):")
                
                for i, weight in enumerate(self.optimal_weights):
                    if weight > 0.01:
                        print(f"  {self.asset_names[i]}: {weight:.4f}")
                        
                return True
            else:
                print("Warning: Could not find optimal portfolio")
                return False
                
        except ImportError:
            print("Warning: scipy not available for optimal portfolio calculation")
            return False
        except Exception as e:
            print(f"Warning: Error calculating optimal portfolio: {e}")
            return False

    def run_experiment(self, 
                      sampling_method: SamplingMethod,
                      time_limit_seconds: int = 60,
                      convergence_threshold: float = 1e-6,
                      patience: int = 500000) -> Dict:
        """Run optimization experiment with given sampling method for a fixed time period."""
        
        print(f"\nRunning experiment with {sampling_method.get_name()} for {time_limit_seconds} seconds...")
        
        best_portfolio = None
        best_score = -np.inf
        history = []
        no_improvement_count = 0
        iteration = 0
        start_time = time.time()
        converged = False
        
        while time.time() - start_time < time_limit_seconds:
            # Generate sample portfolio
            weights = sampling_method.sample_portfolio(self.returns, iteration)
            
            # Predict performance
            performance_score = self.predictor.predict_performance(weights)
            
            # Update Bayesian sampler history if applicable
            if hasattr(sampling_method, 'update_observations'):
                sampling_method.update_observations(weights, performance_score)
            
            # Track best portfolio
            if performance_score > best_score:
                improvement = performance_score - best_score
                best_score = performance_score
                best_portfolio = PortfolioMetrics(weights, performance_score, iteration)
                no_improvement_count = 0
                
                current_time = time.time()
                if iteration % 1000 == 0 or improvement > convergence_threshold:
                    print(f"Iter {iteration} ({current_time - start_time:.1f}s): New best score = {performance_score:.6f}")
            else:
                no_improvement_count += 1
            
            # Get detailed stats for history
            stats = self.predictor.get_portfolio_stats(weights)
            
            # Store history
            history.append({
                'iteration': iteration,
                'performance_score': performance_score,
                'return': stats['return'],
                'volatility': stats['volatility'],
                'sharpe_ratio': stats['sharpe_ratio'],
                'weights': weights.copy(),
                'best_score_so_far': best_score
            })
            
            # Check convergence
            if no_improvement_count >= patience:
                print(f"Converged after {iteration} iterations due to patience.")
                converged = True
                break
            
            iteration += 1

        if not converged:
            print(f"Time limit of {time_limit_seconds} seconds reached. Total iterations: {iteration}.")
        
        results = {
            'method_name': sampling_method.get_name(),
            'best_portfolio': best_portfolio,
            'history': history,
            'total_iterations': len(history),
            'converged': converged,
            'final_best_score': best_score
        }
        
        self.results[sampling_method.get_name()] = results
        return results
    
    def calculate_vector_differences(self):
        """Calculate L2 norm differences between each strategy and optimal"""
        if self.optimal_weights is None:
            print("Optimal portfolio not calculated. Run calculate_optimal_portfolio() first.")
            return {}
        
        differences = {}
        print(f"\nVector Differences from Optimal Portfolio (L2 Norm):")
        print("-" * 60)
        
        for method_name, results in self.results.items():
            best_weights = results['best_portfolio'].weights
            l2_diff = np.linalg.norm(best_weights - self.optimal_weights)
            differences[method_name] = l2_diff
            print(f"{method_name:<30}: {l2_diff:.6f}")
        
        return differences
    
    def plot_portfolio_performance_over_time(self):
        """Plot actual portfolio value over time for all strategies"""
        if self.price_data is None:
            print("Price data not available for performance tracking")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Calculate portfolio values over time
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        # Plot each strategy
        for i, (method_name, results) in enumerate(self.results.items()):
            best_weights = results['best_portfolio'].weights
            
            # Calculate portfolio returns
            portfolio_returns = self.returns.dot(best_weights)
            
            # Calculate cumulative portfolio value (starting at $10,000)
            portfolio_value = (1 + portfolio_returns).cumprod() * 10000
            
            plt.plot(portfolio_value.index, portfolio_value.values, 
                    label=f"{method_name} (Final Sharpe: {results['final_best_score']:.3f})",
                    linewidth=2, color=colors[i % len(colors)])
        
        # Plot optimal portfolio if available
        if self.optimal_weights is not None:
            optimal_returns = self.returns.dot(self.optimal_weights)
            optimal_value = (1 + optimal_returns).cumprod() * 10000
            
            plt.plot(optimal_value.index, optimal_value.values,
                    label=f"Theoretical Optimal (Sharpe: {self.optimal_score:.3f})",
                    linewidth=3, color='black', linestyle='--')
        
        # Plot equal-weight baseline
        equal_weights = np.ones(len(self.asset_names)) / len(self.asset_names)
        equal_returns = self.returns.dot(equal_weights)
        equal_value = (1 + equal_returns).cumprod() * 10000
        equal_sharpe = sharpe_performance_function(equal_weights, self.returns)
        
        plt.plot(equal_value.index, equal_value.values,
                label=f"Equal Weight Baseline (Sharpe: {equal_sharpe:.3f})",
                linewidth=2, color='gray', linestyle=':')
        
        plt.title('Portfolio Performance Over Time', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_allocation_comparison_individual(self):
        """Plot individual allocation charts for each method"""
        if not self.results:
            print("No results to compare")
            return
        
        methods = list(self.results.keys())
        n_methods = len(methods) + (1 if self.optimal_weights is not None else 0) + 1  # +1 for equal weight
        
        # Calculate subplot layout (2 columns)
        n_cols = 2
        n_rows = (n_methods + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        plot_idx = 0
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'plum', 'tan']
        n_assets = len(self.asset_names)
        
        # Plot each strategy
        for i, (method_name, results) in enumerate(self.results.items()):
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]
            
            weights = results['best_portfolio'].weights
            score = results['final_best_score']
            
            bars = ax.bar(range(n_assets), weights, alpha=0.7, color=colors[i % len(colors)])
            ax.set_title(f'{method_name}\n(Sharpe: {score:.4f})', fontsize=10)
            ax.set_ylabel('Weight')
            ax.set_xticks(range(n_assets))
            ax.set_xticklabels(self.asset_names, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on top of bars for weights > 5%
            for j, (bar, weight) in enumerate(zip(bars, weights)):
                if weight > 0.05:  # Only label significant weights
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{weight:.3f}', ha='center', va='bottom', fontsize=7)
            plot_idx += 1
        
        # Plot optimal if available
        if self.optimal_weights is not None:
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]
            
            bars = ax.bar(range(n_assets), self.optimal_weights, alpha=0.7, color='black')
            ax.set_title(f'Theoretical Optimal\n(Sharpe: {self.optimal_score:.4f})', fontsize=10)
            ax.set_ylabel('Weight')
            ax.set_xticks(range(n_assets))
            ax.set_xticklabels(self.asset_names, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add value labels for significant weights
            for j, (bar, weight) in enumerate(zip(bars, self.optimal_weights)):
                if weight > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{weight:.3f}', ha='center', va='bottom', fontsize=7)
            
            plot_idx += 1
        
        # Plot equal weight baseline
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]
        
        equal_weights = np.ones(n_assets) / n_assets
        equal_sharpe = sharpe_performance_function(equal_weights, self.returns)
        
        ax.bar(range(n_assets), equal_weights, alpha=0.7, color='gray')
        ax.set_title(f'Equal Weight Baseline\n(Sharpe: {equal_sharpe:.4f})', fontsize=10)
        ax.set_ylabel('Weight')
        ax.set_xlabel('Assets')
        ax.set_xticks(range(n_assets))
        ax.set_xticklabels(self.asset_names, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
        
        # Hide empty subplots
        for i in range(plot_idx, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_allocation_comparison_stacked(self):
        """Plot stacked allocation comparison showing all methods"""
        if not self.results:
            print("No results to compare")
            return
        
        methods = list(self.results.keys())
        all_methods = methods.copy()
        
        if self.optimal_weights is not None:
            all_methods.append('Theoretical Optimal')
        all_methods.append('Equal Weight')
        
        n_methods = len(all_methods)
        n_assets = len(self.asset_names)
        
        # Prepare data
        weights_matrix = np.zeros((n_methods, n_assets))
        method_labels = []
        sharpe_scores = []
        
        # Fill in strategy weights
        for i, method_name in enumerate(methods):
            weights_matrix[i, :] = self.results[method_name]['best_portfolio'].weights
            method_labels.append(method_name)
            sharpe_scores.append(self.results[method_name]['final_best_score'])
        
        idx = len(methods)
        
        # Add optimal if available
        if self.optimal_weights is not None:
            weights_matrix[idx, :] = self.optimal_weights
            method_labels.append('Theoretical Optimal')
            sharpe_scores.append(self.optimal_score)
            idx += 1
        
        # Add equal weight
        equal_weights = np.ones(n_assets) / n_assets
        equal_sharpe = sharpe_performance_function(equal_weights, self.returns)
        weights_matrix[idx, :] = equal_weights
        method_labels.append('Equal Weight')
        sharpe_scores.append(equal_sharpe)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Create stacked bar chart
        colors = plt.cm.Set3(np.linspace(0, 1, n_assets))
        bottom = np.zeros(n_methods)
        
        for asset_idx in range(n_assets):
            ax.bar(range(n_methods), weights_matrix[:, asset_idx], 
                   bottom=bottom, label=self.asset_names[asset_idx], 
                   color=colors[asset_idx], alpha=0.8)
            bottom += weights_matrix[:, asset_idx]
        
        ax.set_xlabel('Methods')
        ax.set_ylabel('Portfolio Weight')
        ax.set_title('Portfolio Allocation Comparison (Stacked)')
        
        # Set x-axis labels with Sharpe ratios
        x_labels = [f'{method}\n(Sharpe: {score:.3f})' for method, score in zip(method_labels, sharpe_scores)]
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
        # Legend outside plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.show()
    
    def print_comprehensive_summary(self):
        """Print comprehensive analysis summary"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE PORTFOLIO OPTIMIZATION ANALYSIS")
        print(f"{'='*80}")
        
        print(f"\nDataset Information:")
        print(f"Assets: {len(self.asset_names)}")
        print(f"Time periods: {len(self.returns)}")
        
        # Handle date range display for both real and simulated data
        try:
            if hasattr(self.returns.index[0], 'strftime'):
                print(f"Date range: {self.returns.index[0].strftime('%Y-%m-%d')} to {self.returns.index[-1].strftime('%Y-%m-%d')}")
            else:
                print(f"Index range: {self.returns.index[0]} to {self.returns.index[-1]}")
        except:
            print("Date range: Simulated data")
        
        # Calculate equal weight baseline
        equal_weights = np.ones(len(self.asset_names)) / len(self.asset_names)
        equal_sharpe = sharpe_performance_function(equal_weights, self.returns)
        
        print(f"\nBaseline Performance:")
        print(f"Equal Weight Sharpe Ratio: {equal_sharpe:.4f}")
        
        if self.optimal_weights is not None:
            print(f"Theoretical Optimal Sharpe: {self.optimal_score:.4f}")
        
        print(f"\nStrategy Results:")
        print(f"{'Method':<30} {'Sharpe':<8} {'vs Equal':<10} {'vs Optimal':<12} {'L2 Diff':<10}")
        print("-" * 80)
        
        for method_name, results in self.results.items():
            sharpe = results['final_best_score']
            vs_equal = sharpe - equal_sharpe
            vs_optimal = (sharpe - self.optimal_score) if self.optimal_score else "N/A"
            
            # Calculate L2 difference
            if self.optimal_weights is not None:
                l2_diff = np.linalg.norm(results['best_portfolio'].weights - self.optimal_weights)
                l2_str = f"{l2_diff:.6f}"
            else:
                l2_str = "N/A"
            
            vs_opt_str = f"{vs_optimal:.4f}" if vs_optimal != "N/A" else "N/A"
            
            print(f"{method_name:<30} {sharpe:<8.4f} {vs_equal:<10.4f} {vs_opt_str:<12} {l2_str:<10}")

# Example usage and testing
if __name__ == "__main__":
    # Create test environment
    env = TestEnvironment(
        n_assets=15,
        performance_function=sharpe_performance_function,
        use_real_data=True
    )
    
    # Calculate optimal portfolio
    print("Calculating theoretical optimal portfolio...")
    optimal_found = env.calculate_optimal_portfolio()
    
    # Define sampling methods
    samplers = [
        RandomUniformSampling(),
        FastHierarchicalSampling(equal_weight_bias=0.2),
        FastHierarchicalSampling(equal_weight_bias=0.7),
        HierarchicalBayesianSampling(equal_weight_bias=0.2, acquisition_function='pi')
    ]
    
    # Run experiments
    for sampler in samplers:
        print(f"\n{'='*60}")
        print(f"Testing {sampler.get_name()}")
        print(f"{'='*60}")
        
        results = env.run_experiment(
            sampling_method=sampler,
            time_limit_seconds=10,
            patience=20000
        )
    
    # Comprehensive analysis
    env.print_comprehensive_summary()
    
    # Calculate vector differences
    if optimal_found:
        env.calculate_vector_differences()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # NOTE: The original plotting functions may not exist. 
    # The following calls are based on the methods present in the class definition above.
    
    # Plot allocations - individual plots
    print("3. Individual allocation plots...")
    env.plot_allocation_comparison_individual()
    
    # Plot allocations - stacked comparison
    print("4. Stacked allocation comparison...")
    env.plot_allocation_comparison_stacked()
    
    # Plot portfolio performance over time
    print("5. Portfolio performance over time...")
    env.plot_portfolio_performance_over_time()