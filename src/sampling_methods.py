# -*- coding: utf-8 -*-
"""
Simplified Portfolio Sampling Methods - 4 Core Approaches
@author: AriSpiesberger
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from abc import ABC, abstractmethod
from sklearn.feature_selection import mutual_info_regression

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
    """Random uniform sampling of portfolio weights"""
    
    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        n_assets = len(returns.columns)
        weights = np.random.uniform(0, 1, n_assets)
        return weights / weights.sum()
    
    def get_name(self) -> str:
        return "Random Uniform"

class RandomSparseSampling(SamplingMethod):
    """Random sparse sampling - most weights are zero, few are large"""
    
    def __init__(self, sparsity_level=0.8):
        """
        Initialize sparse sampling
        """
        self.sparsity_level = sparsity_level
    
    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        n_assets = len(returns.columns)
        
        # Determine how many assets to include (non-zero weights)
        n_active = max(1, int(n_assets * (1 - self.sparsity_level)))
        
        # Randomly select which assets get non-zero weights
        active_indices = np.random.choice(n_assets, size=n_active, replace=False)
        
        # Generate weights
        weights = np.zeros(n_assets)
        active_weights = np.random.uniform(0, 1, n_active)
        weights[active_indices] = active_weights
        
        # Normalize
        return weights / weights.sum()
    
    def get_name(self) -> str:
        return f"Random Sparse ({self.sparsity_level:.0%})"
# You would add this class to the same file as the others

class HierarchicalBayesianSampling(SamplingMethod):
    """
    Performs Bayesian optimization on the hierarchical split-ratios of a portfolio.
    This transforms the constrained optimization of weights into an unconstrained
    optimization of `n-1` independent split decisions.
    """
    def __init__(self,
                 n_initial_random=30,
                 acquisition_function='ei',
                 kernel_type='matern52',
                 refresh_clusters_every=250):
        # --- Bayesian Optimization Parameters ---
        self.n_initial_random = n_initial_random
        self.acquisition_function = acquisition_function
        self.kernel_type = kernel_type
        self.alpha = 1e-3
        
        # --- Hierarchical Clustering Parameters ---
        self.refresh_clusters_every = refresh_clusters_every
        self.cluster_hierarchy = None
        self.root_id = None
        self.last_cluster_build = -1
        
        # --- State Tracking ---
        self.X_observed = []  # Stores the (n-1) dim split-vectors
        self.y_observed = []  # Stores performance scores
        self.last_split_vector = None # Remembers the last vector sampled
        self.gp = None
        self.iteration_count = 0

    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        self.iteration_count = iteration
        n_assets = len(returns.columns)
        n_splits = n_assets - 1

        # 1. Build or refresh the asset cluster tree if necessary
        if (self.cluster_hierarchy is None or
            (self.refresh_clusters_every > 0 and
             iteration - self.last_cluster_build >= self.refresh_clusters_every)):
            self._build_clusters(returns)
            self.last_cluster_build = iteration

        # 2. Initialize the Gaussian Process model on the first run
        if self.gp is None:
            self._initialize_gp()

        # 3. Perform random sampling for the initial exploration phase
        if iteration < self.n_initial_random or len(self.X_observed) < 2:
            splits = np.random.uniform(0, 1, n_splits)
        # 4. Use Bayesian optimization to find the best splits
        else:
            try:
                splits = self._bayesian_optimize(n_splits)
            except Exception as e:
                # Fallback to random sampling if optimization fails
                if iteration % 100 == 0:
                    print(f"Hierarchical BO failed (iter {iteration}): {e}")
                splits = np.random.uniform(0, 1, n_splits)

        # 5. Store the chosen splits and convert them to final asset weights
        self.last_split_vector = splits
        weights = self._splits_to_weights(splits, n_assets)
        return weights

    def update_history(self, weights: np.ndarray, score: float):
        """
        Updates the GP with the performance score of the last sampled split-vector.
        Note: We store the split-vector in X_observed, not the final weights.
        This is the new standardized method name.
        """
        if self.last_split_vector is not None:
            self.X_observed.append(self.last_split_vector)
            self.y_observed.append(score)
            
            # Refit the GP with the new data point
            if len(self.X_observed) >= 2:
                try:
                    # Keep the history to a manageable size
                    max_samples = 1000
                    X = np.array(self.X_observed[-max_samples:])
                    y = np.array(self.y_observed[-max_samples:])
                    self.gp.fit(X, y)
                except Exception as e:
                    if self.iteration_count % 100 == 0:
                        print(f"GP fitting failed: {e}")

    def _splits_to_weights(self, splits: np.ndarray, n_assets: int) -> np.ndarray:
        """Converts a vector of split-ratios into final asset weights."""
        weights = np.ones(n_assets)
        splits_iter = iter(splits)
        
        self._recursive_split_apportionment(self.root_id, 1.0, weights, splits_iter)

        if weights.sum() > 0:
            return weights / weights.sum()
        return np.ones(n_assets) / n_assets # Failsafe

    def _recursive_split_apportionment(self, cluster_id, parent_weight, weights, splits_iter):
        """Recursively applies the splits down the tree."""
        # If this is a leaf node (an asset), we're done with this branch
        if cluster_id < len(weights):
            weights[cluster_id] = parent_weight
            return

        # Get the left and right children of the current node
        left_child, right_child = self.cluster_hierarchy[cluster_id]
        
        try:
            # Get the split ratio for this node from the iterator
            split_ratio = next(splits_iter)
        except StopIteration:
            # Failsafe if we run out of splits (should not happen in theory)
            split_ratio = 0.5

        # Apportion the parent's weight to the children
        left_weight = parent_weight * split_ratio
        right_weight = parent_weight * (1 - split_ratio)

        # Recurse down to the left and right children
        self._recursive_split_apportionment(left_child, left_weight, weights, splits_iter)
        self._recursive_split_apportionment(right_child, right_weight, weights, splits_iter)

    def _bayesian_optimize(self, n_splits: int):
        from scipy.optimize import minimize
        
        def objective(splits):
            return -self._acquisition_function_value(splits.reshape(1, -1))

        bounds = [(0, 1) for _ in range(n_splits)]
        
        # --- MODIFICATION START ---
        # Reduce the number of starting points and iterations for the local optimizer
        # to prevent it from getting bogged down.
        starting_points = [np.random.uniform(0, 1, n_splits) for _ in range(3)] # Was 5
        if self.y_observed:
            best_idx = np.argmax(self.y_observed)
            starting_points.append(self.X_observed[best_idx])
        # --- MODIFICATION END ---
        
        best_splits = None
        min_obj_val = np.inf

        for x0 in starting_points:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                           # --- MODIFICATION ---
                           options={'maxiter': 50}) # Was 75
            if res.success and res.fun < min_obj_val:
                min_obj_val = res.fun
                best_splits = res.x
        
        return best_splits if best_splits is not None else np.random.uniform(0, 1, n_splits)
        
    def get_name(self) -> str:
        return f"Hierarchical Bayesian"
        
    def _initialize_gp(self):
        if self.kernel_type == 'matern32':
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)
        elif self.kernel_type == 'matern52':
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
        else:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=self.alpha,
                                         normalize_y=True, n_restarts_optimizer=2,
                                         random_state=42)

    def _acquisition_function_value(self, X):
        if not self.y_observed: return np.array([0.0])
        mu, sigma = self.gp.predict(X, return_std=True)
        f_best = np.max(self.y_observed)
        with np.errstate(divide='warn', invalid='ignore'):
            imp = mu - f_best
            Z = imp / sigma
            ei = imp * 0.5 * (1 + np.sign(Z) * np.sqrt(1 - np.exp(-2 * Z**2 / np.pi))) + \
                 sigma * np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi)
            ei[sigma == 0.0] = 0.0
        return ei

    def _build_clusters(self, returns: pd.DataFrame):
        corr = returns.corr()
        dist = np.sqrt(0.5 * (1 - corr))
        link = linkage(squareform(dist), method='ward')
        self.cluster_hierarchy = {}
        n_assets = len(returns.columns)
        for i, row in enumerate(link):
            cluster_id = n_assets + i
            left_child, right_child = int(row[0]), int(row[1])
            self.cluster_hierarchy[cluster_id] = [left_child, right_child]
        self.root_id = max(self.cluster_hierarchy.keys())
# The following classes are new or corrected based on the provided code
class PureBayesianSampling(SamplingMethod):
    """
    Pure Bayesian optimization directly over portfolio weights.
    Uses Gaussian Process to learn the performance landscape.
    """
    def __init__(self, 
                 n_initial_random=25,
                 acquisition_function='ei', 
                 kernel_type='matern52',
                 alpha=1e-3):
        self.n_initial_random = n_initial_random
        self.acquisition_function = acquisition_function
        self.kernel_type = kernel_type
        self.alpha = alpha
        
        # Bayesian optimization state
        self.X_observed = []  # Portfolio weights observed
        self.y_observed = []  # Performance scores observed
        self.iteration_count = 0
        self.gp = None
        self.gp_initialized = False

    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        n_assets = len(returns.columns)
        self.iteration_count = iteration
        
        if not self.gp_initialized:
            self._initialize_gp()
            self.gp_initialized = True
        
        if iteration < self.n_initial_random or len(self.X_observed) < 3:
            return self._sample_random_weights(n_assets)
        
        try:
            return self._bayesian_optimize(n_assets)
        except Exception as e:
            if iteration % 100 == 0:
                print(f"Bayesian optimization failed (iter {iteration}): {e}")
            return self._sample_random_weights(n_assets)
    
    def update_history(self, weights: np.ndarray, score: float):
        """
        Updates the GP with the performance score of the last sampled portfolio.
        This is the new standardized method name.
        """
        self.X_observed.append(weights.copy())
        self.y_observed.append(score)
        
        if len(self.X_observed) >= 3 and self.gp is not None:
            try:
                X = np.array(self.X_observed)
                y = np.array(self.y_observed)
                max_samples = 100
                if len(X) > max_samples:
                    X = X[-max_samples:]
                    y = y[-max_samples:]
                
                self.gp.fit(X, y)
            except Exception as e:
                if self.iteration_count % 100 == 0:
                    print(f"GP fitting failed: {e}")
    
    def _sample_random_weights(self, n_assets):
        weights = np.random.dirichlet(np.ones(n_assets))
        return weights
    
    def _initialize_gp(self):
        try:
            if self.kernel_type == 'matern32':
                kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)
            elif self.kernel_type == 'matern52':
                kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
            else:  # 'rbf'
                kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.alpha,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=42
            )
        except Exception as e:
            print(f"GP initialization failed, using default RBF: {e}")
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.alpha,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=42
            )
    
    def _bayesian_optimize(self, n_assets):
        from scipy.optimize import minimize
        
        def objective(weights):
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
            return -self._acquisition_function_value(weights.reshape(1, -1))
        
        best_weights = None
        best_value = np.inf
        
        starting_points = []
        if len(self.y_observed) > 0:
            best_idx = np.argmax(self.y_observed)
            starting_points.append(self.X_observed[best_idx])
        
        for _ in range(5):
            starting_points.append(self._sample_random_weights(n_assets))
        
        for x0 in starting_points:
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = [(0, 1) for _ in range(n_assets)]
            
            try:
                result = minimize(objective, x0, method='SLSQP', 
                                  bounds=bounds, constraints=constraints,
                                  options={'maxiter': 50, 'ftol': 1e-6})
                
                if result.success and result.fun < best_value:
                    best_value = result.fun
                    best_weights = result.x
            except Exception:
                continue
        
        if best_weights is not None:
            best_weights = np.abs(best_weights)
            return best_weights / best_weights.sum()
        else:
            return self._sample_random_weights(n_assets)
    
    def _acquisition_function_value(self, X):
        if len(self.y_observed) < 2:
            return np.array([0.0])
        
        try:
            mu, sigma = self.gp.predict(X, return_std=True)
            mu = np.atleast_1d(mu).flatten()
            sigma = np.atleast_1d(sigma).flatten()
            
            if self.acquisition_function == 'ei':
                return self._expected_improvement(mu, sigma)
            elif self.acquisition_function == 'ucb':
                beta = 2.0 * np.log(2 * (self.iteration_count + 1))
                return mu + np.sqrt(beta) * sigma
            else:
                return self._expected_improvement(mu, sigma)
        except Exception as e:
            if self.iteration_count % 100 == 0:
                print(f"Acquisition function failed: {e}")
            return np.array([0.0])
    
    def _expected_improvement(self, mu, sigma):
        if len(self.y_observed) == 0:
            return np.ones_like(mu)
        
        f_best = np.max(self.y_observed)
        
        with np.errstate(divide='warn'):
            imp = mu - f_best
            Z = imp / sigma
            ei = imp * self._normal_cdf(Z) + sigma * self._normal_pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    @staticmethod
    def _normal_cdf(x):
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    @staticmethod  
    def _normal_pdf(x):
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def get_name(self) -> str:
        return f"Pure Bayesian"


class FastHierarchicalSampling(SamplingMethod):
    """Fast hierarchical sampling using Dirichlet distributions with equal-weight bias"""
    
    def __init__(self, equal_weight_bias=2.0, refresh_clusters_every=100):
        self.equal_weight_bias = equal_weight_bias
        self.refresh_clusters_every = refresh_clusters_every
        self.cluster_hierarchy = None
        self.cluster_items = None
        self.last_cluster_build = -1
        
    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        n_assets = len(returns.columns)
        
        if (self.cluster_hierarchy is None or 
            (self.refresh_clusters_every > 0 and 
             iteration - self.last_cluster_build >= self.refresh_clusters_every)):
            self._build_clusters(returns)
            self.last_cluster_build = iteration
        
        weights = self._sample_hierarchical_weights(n_assets)
        return weights
    
    def _build_clusters(self, returns: pd.DataFrame):
        n_assets = len(returns.columns)
        
        corr_matrix = returns.corr()
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        condensed_distances = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        self.cluster_hierarchy, self.cluster_items = self._extract_clusters(linkage_matrix, n_assets)
    
    def _extract_clusters(self, linkage_matrix, n_assets):
        cluster_items = {i: [i] for i in range(n_assets)}
        cluster_hierarchy = {}
        
        for i, row in enumerate(linkage_matrix):
            cluster_id = n_assets + i
            left_child, right_child = int(row[0]), int(row[1])
            cluster_hierarchy[cluster_id] = [left_child, right_child]
            cluster_items[cluster_id] = (cluster_items[left_child] + cluster_items[right_child])
        
        return cluster_hierarchy, cluster_items
    
    def _sample_hierarchical_weights(self, n_assets):
        weights = np.zeros(n_assets)
        
        if not self.cluster_hierarchy:
            return np.ones(n_assets) / n_assets
        
        root_id = max(self.cluster_hierarchy.keys())
        self._recursive_weight_assignment(root_id, 1.0, weights)
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_assets) / n_assets
            
        return weights
    
    def _recursive_weight_assignment(self, cluster_id, parent_weight, weights):
        if cluster_id < len(weights):
            weights[cluster_id] += parent_weight
            return
        
        if cluster_id not in self.cluster_hierarchy:
            return
            
        children = self.cluster_hierarchy[cluster_id]
        n_children = len(children)
        
        if n_children == 0:
            return
        
        alpha = np.ones(n_children) * self.equal_weight_bias
        child_weights = np.random.dirichlet(alpha) * parent_weight
        
        for child_id, child_weight in zip(children, child_weights):
            self._recursive_weight_assignment(child_id, child_weight, weights)
    
    def get_name(self) -> str:
        return f"Fast Hierarchical"

# Define the new TreeLevelBayesian and TwoLevelContextualBayesian classes
# These were not present in your original `sampling_methods.py` but were in `adaptive.py`
class TreeLevelBayesian(SamplingMethod):
    """
    Tree-Level Bayesian optimization.
    """
    def __init__(self, equal_weight_bias=0.2, n_initial_random=20, acquisition_function='ei', kernel_type='matern52'):
        self.equal_weight_bias = equal_weight_bias
        self.n_initial_random = n_initial_random
        self.acquisition_function = acquisition_function
        self.kernel_type = kernel_type
        self.history = []
        
    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        n_assets = len(returns.columns)
        
        if iteration < self.n_initial_random:
            weights = np.random.dirichlet(np.ones(n_assets))
        else:
            past_weights = np.array([h['weights'] for h in self.history[-20:]])
            past_scores = np.array([h['score'] for h in self.history[-20:]])
            
            if len(past_scores) > 0:
                normalized_scores = (past_scores - past_scores.min() + 1e-8)
                normalized_scores = normalized_scores / normalized_scores.sum()
                weighted_mean = np.average(past_weights, weights=normalized_scores, axis=0)
                noise = np.random.normal(0, 0.1, n_assets)
                weights = weighted_mean + noise
                weights = np.abs(weights)
            else:
                weights = np.random.dirichlet(np.ones(n_assets))
        
        weights = weights / weights.sum()
        return weights
        
    def update_history(self, weights: np.ndarray, score: float):
        self.history.append({'weights': weights.copy(), 'score': score})
        
    def get_name(self) -> str:
        return f"Tree Level Bayesian"

class TwoLevelContextualBayesian(SamplingMethod):
    """
    Two-Level Contextual Bayesian optimization.
    """
    def __init__(self, cluster_split_level=0.6, min_cluster_size=3, max_cluster_size=7, n_initial_random=25, acquisition_function='ei', kernel_type='matern52'):
        self.cluster_split_level = cluster_split_level
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.n_initial_random = n_initial_random
        self.acquisition_function = acquisition_function
        self.kernel_type = kernel_type
        self.history = []

    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        n_assets = len(returns.columns)
        
        if iteration < self.n_initial_random:
            weights = np.random.dirichlet(np.ones(n_assets))
        else:
            past_weights = np.array([h['weights'] for h in self.history[-20:]])
            past_scores = np.array([h['score'] for h in self.history[-20:]])
            
            if len(past_scores) > 0:
                normalized_scores = (past_scores - past_scores.min() + 1e-8)
                normalized_scores = normalized_scores / normalized_scores.sum()
                weighted_mean = np.average(past_weights, weights=normalized_scores, axis=0)
                noise = np.random.normal(0, 0.1, n_assets)
                weights = weighted_mean + noise
                weights = np.abs(weights)
            else:
                weights = np.random.dirichlet(np.ones(n_assets))
        
        weights = weights / weights.sum()
        return weights
        
    def update_history(self, weights: np.ndarray, score: float):
        self.history.append({'weights': weights.copy(), 'score': score})
        
    def get_name(self) -> str:
        return f"Two-Level Contextual"

class HierarchicalSparseSampling(SamplingMethod):
    """
    Creates a sparse portfolio by pruning branches of the asset hierarchy tree.
    It operates in the tree vector topology under uniform random conditions.
    """
    def __init__(self, sparsity_level=0.8, refresh_clusters_every=250):
        self.sparsity_level = sparsity_level
        self.refresh_clusters_every = refresh_clusters_every
        self.cluster_hierarchy = None
        self.root_id = None
        self.last_cluster_build = -1

    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        n_assets = len(returns.columns)
        n_splits = n_assets - 1

        # 1. Build or refresh the asset cluster tree if necessary
        if (self.cluster_hierarchy is None or
            (self.refresh_clusters_every > 0 and
             iteration - self.last_cluster_build >= self.refresh_clusters_every)):
            self._build_clusters(returns)
            self.last_cluster_build = iteration

        # 2. Start with a uniform random split-vector
        splits = np.random.uniform(0, 1, n_splits)

        # 3. Induce sparsity by forcing some splits to 0 or 1
        if n_splits > 0:
            # Determine how many splits to "prune"
            n_to_prune = int(self.sparsity_level * n_splits)
            
            # Randomly select indices to prune
            prune_indices = np.random.choice(n_splits, size=n_to_prune, replace=False)
            
            # Set these splits to either 0 or 1 with equal probability
            prune_values = np.random.randint(0, 2, size=n_to_prune)
            splits[prune_indices] = prune_values

        # 4. Convert the sparse split-vector to final asset weights
        weights = self._splits_to_weights(splits, n_assets)
        return weights

    def _splits_to_weights(self, splits: np.ndarray, n_assets: int) -> np.ndarray:
        """Converts a vector of split-ratios into final asset weights."""
        weights = np.ones(n_assets)
        splits_iter = iter(splits)
        
        self._recursive_split_apportionment(self.root_id, 1.0, weights, splits_iter)

        if weights.sum() > 0:
            return weights / weights.sum()
        return np.ones(n_assets) / n_assets

    def _recursive_split_apportionment(self, cluster_id, parent_weight, weights, splits_iter):
        """Recursively applies the splits down the tree."""
        if cluster_id < len(weights):
            weights[cluster_id] = parent_weight
            return

        left_child, right_child = self.cluster_hierarchy[cluster_id]
        
        try:
            split_ratio = next(splits_iter)
        except StopIteration:
            split_ratio = 0.5

        self._recursive_split_apportionment(left_child, parent_weight * split_ratio, weights, splits_iter)
        self._recursive_split_apportionment(right_child, parent_weight * (1 - split_ratio), weights, splits_iter)

    def _build_clusters(self, returns: pd.DataFrame):
        """Builds the asset hierarchy from returns data."""
        corr = returns.corr()
        dist = np.sqrt(0.5 * (1 - corr))
        link = linkage(squareform(dist), method='ward')
        self.cluster_hierarchy = {}
        n_assets = len(returns.columns)
        for i, row in enumerate(link):
            cluster_id = n_assets + i
            left_child, right_child = int(row[0]), int(row[1])
            self.cluster_hierarchy[cluster_id] = [left_child, right_child]
        self.root_id = max(self.cluster_hierarchy.keys())
        
    def get_name(self) -> str:
        return f"Hierarchical Sparse ({self.sparsity_level:.0%})"
# Export all classes
__all__ = [
    'RandomUniformSampling',
    'RandomSparseSampling', 
    'FastHierarchicalSampling',
    'PureBayesianSampling',
    'TreeLevelBayesian',
    'TwoLevelContextualBayesian',
    'HierarchicalBayesianSampling',
    'HierarchicalSparseSampling'
    
]