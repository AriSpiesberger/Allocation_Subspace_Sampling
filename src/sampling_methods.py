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
        
        Args:
            sparsity_level: Fraction of assets to set to zero (0.8 = 80% sparse)
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



import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sampling_methods import SamplingMethod
class FlowHierarchicalSampling(SamplingMethod):
    """Sparse hierarchical sampling with minimal memory allocation."""
    
    def __init__(self, 
                 n_paths: int = 50, 
                 lr: float = 0.1, 
                 baseline_decay: float = 0.9, 
                 refresh_clusters_every: int = 10000):
        self.n_paths = n_paths
        self.lr = lr
        self.baseline_decay = baseline_decay
        self.refresh_clusters_every = refresh_clusters_every
        
        # Sparse representations
        self.n_assets = None
        self.asset_mask = None           # Boolean mask for valid assets
        self.valid_asset_map = None      # Map from original to valid indices
        self.tree_structure = None       # Sparse tree: {internal_node: [child1, child2]}
        self.internal_nodes = None       # Set of internal node IDs
        self.theta = {}                  # Sparse: {internal_node: [logit1, logit2]}
        self.node_counts = {}            # Sparse: {internal_node: [count1, count2]}
        
        self.baseline = None
        self.last_cluster_build = -1

    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        # Rebuild tree if needed
        if (self.tree_structure is None or
            (self.refresh_clusters_every > 0 and iteration - self.last_cluster_build >= self.refresh_clusters_every)):
            self._build_sparse_tree(returns.values)
            self._initialize_sparse_flow()
            self.last_cluster_build = iteration
        
        # Initialize portfolio weights
        portfolio_weights = np.zeros(self.n_assets)
        
        # Reset node counts for this sample
        self.node_counts = {node: np.zeros(2) for node in self.internal_nodes}
        
        # Sample multiple paths
        for _ in range(self.n_paths):
            # Start from root (highest internal node ID)
            current_node = max(self.internal_nodes)
            
            # Traverse down the tree
            while current_node in self.internal_nodes:
                # Get logits for current node
                logits = self.theta[current_node]
                # Compute probabilities (numerically stable)
                max_logit = np.max(logits)
                exp_logits = np.exp(logits - max_logit)
                probs = exp_logits / np.sum(exp_logits)
                
                # Sample child
                child_idx = np.random.choice(2, p=probs)
                self.node_counts[current_node][child_idx] += 1
                
                # Move to selected child
                current_node = self.tree_structure[current_node][child_idx]
            
            # Current node is now a leaf (asset index)
            if current_node < self.n_assets:
                portfolio_weights[current_node] += 1.0
        
        # Normalize weights
        return portfolio_weights / self.n_paths

    def update_observations(self, weights: np.ndarray, performance: float):
        """Update using REINFORCE gradient."""
        # Update baseline
        if self.baseline is None:
            self.baseline = performance
        else:
            self.baseline = self.baseline * self.baseline_decay + performance * (1 - self.baseline_decay)
        
        advantage = performance - self.baseline
        
        # Update each internal node
        for node_id in self.internal_nodes:
            if node_id not in self.node_counts:
                continue
                
            logits = self.theta[node_id]
            counts = self.node_counts[node_id]
            
            # Compute current probabilities
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)
            
            # REINFORCE gradient: advantage * (empirical_prob - current_prob)
            empirical_probs = counts / self.n_paths
            gradient = advantage * (empirical_probs - probs)
            
            # Update logits
            self.theta[node_id] += self.lr * gradient

    def get_name(self) -> str:
        return f"SparseFlowHierarchical(paths={self.n_paths}, lr={self.lr:.2f})"

    def _build_sparse_tree(self, returns_array: np.ndarray):
        """Build sparse tree representation."""
        original_n_assets = returns_array.shape[1]
        
        # Filter valid assets (non-zero std dev)
        stds = np.std(returns_array, axis=0)
        self.asset_mask = stds > 1e-8
        valid_returns = returns_array[:, self.asset_mask]
        n_valid = valid_returns.shape[1]
        
        # Update n_assets to match the actual number of assets we're working with
        self.n_assets = original_n_assets  # Keep original for weight vector size
        
        if n_valid < 2:
            raise ValueError("Need at least 2 valid assets for hierarchical clustering")
        
        # Create mapping from valid asset indices to original indices
        valid_indices = np.where(self.asset_mask)[0]
        self.valid_asset_map = {i: valid_indices[i] for i in range(n_valid)}
        
        # Compute correlation matrix for valid assets only
        if valid_returns.shape[0] < 2:
            # Not enough data points, create simple binary tree
            self.tree_structure = {}
            self.internal_nodes = set()
            
            # Create a simple binary split
            mid = n_valid // 2
            root_id = n_valid  # First internal node ID
            self.tree_structure[root_id] = [0, 1 if n_valid > 1 else 0]
            self.internal_nodes.add(root_id)
            
            print(f"Warning: Limited data, created simple tree with {len(self.internal_nodes)} internal nodes")
            return
        
        try:
            # Build correlation-based distance matrix
            corr = np.corrcoef(valid_returns, rowvar=False)
            np.fill_diagonal(corr, 1.0)
            corr = np.clip(corr, -1 + 1e-6, 1 - 1e-6)
            
            # Convert to distance
            dist = np.sqrt(0.5 * (1 - corr))
            
            # Hierarchical clustering
            from scipy.spatial.distance import squareform
            from scipy.cluster.hierarchy import linkage
            
            condensed_dist = squareform(dist, checks=False)
            
            if not np.all(np.isfinite(condensed_dist)):
                raise ValueError("Distance matrix contains non-finite values")
            
            linkage_matrix = linkage(condensed_dist, method='ward')
            
            # Build sparse tree structure
            self.tree_structure = {}
            self.internal_nodes = set()
            
            # Internal nodes start after leaf nodes
            internal_offset = n_valid
            
            for i, (left, right, _, _) in enumerate(linkage_matrix):
                internal_id = internal_offset + i
                
                # Map cluster indices back to original asset indices if they're leaves
                left_child = valid_indices[int(left)] if left < n_valid else int(left)
                right_child = valid_indices[int(right)] if right < n_valid else int(right)
                
                self.tree_structure[internal_id] = [left_child, right_child]
                self.internal_nodes.add(internal_id)
            
            print(f"Built sparse tree: {n_valid} valid assets, {len(self.internal_nodes)} internal nodes")
            
        except Exception as e:
            print(f"Warning: Clustering failed ({e}), creating simple binary tree")
            # Fallback: create simple binary tree
            self._create_simple_binary_tree(n_valid)

    def _create_simple_binary_tree(self, n_valid: int):
        """Create a simple binary tree as fallback."""
        self.tree_structure = {}
        self.internal_nodes = set()
        
        if n_valid == 1:
            return  # Single asset, no internal nodes needed
        
        # Create binary tree recursively
        def build_binary_subtree(asset_indices, next_internal_id):
            if len(asset_indices) == 1:
                return asset_indices[0], next_internal_id
            elif len(asset_indices) == 2:
                # Create internal node for two assets
                internal_id = next_internal_id
                mapped_left = valid_indices[asset_indices[0]]
                mapped_right = valid_indices[asset_indices[1]]
                self.tree_structure[internal_id] = [mapped_left, mapped_right]
                self.internal_nodes.add(internal_id)
                return internal_id, next_internal_id + 1
            else:
                # Split in half
                mid = len(asset_indices) // 2
                left_indices = asset_indices[:mid]
                right_indices = asset_indices[mid:]
                
                left_root, next_id = build_binary_subtree(left_indices, next_internal_id)
                right_root, next_id = build_binary_subtree(right_indices, next_id)
                
                # Create parent internal node
                parent_id = next_id
                self.tree_structure[parent_id] = [left_root, right_root]
                self.internal_nodes.add(parent_id)
                
                return parent_id, next_id + 1
        
        # Build tree starting from all valid asset indices
        asset_indices = list(range(n_valid))
        build_binary_subtree(asset_indices, n_valid)

    def _initialize_sparse_flow(self):
        """Initialize flow parameters sparsely."""
        self.theta = {}
        self.node_counts = {}
        
        # Only initialize parameters for internal nodes that exist
        for node_id in self.internal_nodes:
            self.theta[node_id] = np.zeros(2)  # Two children per internal node
            self.node_counts[node_id] = np.zeros(2)
        
        self.baseline = None
        print(f"Initialized sparse flow: {len(self.theta)} internal nodes")



class FastHierarchicalSampling(SamplingMethod):
    """Fast hierarchical sampling using Dirichlet distributions with equal-weight bias"""
    
    def __init__(self, equal_weight_bias=2.0, refresh_clusters_every=100):
        """
        Initialize hierarchical sampling
        
        Args:
            equal_weight_bias: Higher values favor equal weights within clusters
            refresh_clusters_every: How often to rebuild clusters (0 = never refresh)
        """
        self.equal_weight_bias = equal_weight_bias
        self.refresh_clusters_every = refresh_clusters_every
        self.cluster_hierarchy = None
        self.cluster_items = None
        self.last_cluster_build = -1
        
    def sample_portfolio(self, returns: pd.DataFrame, iteration: int) -> np.ndarray:
        n_assets = len(returns.columns)
        
        # Build or refresh clusters periodically
        if (self.cluster_hierarchy is None or 
            (self.refresh_clusters_every > 0 and 
             iteration - self.last_cluster_build >= self.refresh_clusters_every)):
            self._build_clusters(returns)
            self.last_cluster_build = iteration
        
        weights = self._sample_hierarchical_weights(n_assets)
        return weights
    
    def _build_clusters(self, returns: pd.DataFrame):
        """Build hierarchical clusters from correlation matrix"""
        n_assets = len(returns.columns)
        
        # Build correlation-based distance matrix
        corr_matrix = returns.corr()
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        condensed_distances = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        # Extract cluster hierarchy
        self.cluster_hierarchy, self.cluster_items = self._extract_clusters(linkage_matrix, n_assets)
    
    def _extract_clusters(self, linkage_matrix, n_assets):
        """Extract cluster hierarchy from linkage matrix"""
        cluster_items = {i: [i] for i in range(n_assets)}
        cluster_hierarchy = {}
        
        for i, row in enumerate(linkage_matrix):
            cluster_id = n_assets + i
            left_child, right_child = int(row[0]), int(row[1])
            cluster_hierarchy[cluster_id] = [left_child, right_child]
            cluster_items[cluster_id] = (cluster_items[left_child] + cluster_items[right_child])
        
        return cluster_hierarchy, cluster_items
    
    def _sample_hierarchical_weights(self, n_assets):
        """Sample portfolio weights using hierarchical structure"""
        weights = np.zeros(n_assets)
        
        if not self.cluster_hierarchy:
            return np.ones(n_assets) / n_assets
        
        root_id = max(self.cluster_hierarchy.keys())
        self._recursive_weight_assignment(root_id, 1.0, weights)
        
        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_assets) / n_assets
            
        return weights
    
    def _recursive_weight_assignment(self, cluster_id, parent_weight, weights):
        """Recursively assign weights through the hierarchy"""
        if cluster_id < len(weights):
            # Leaf node (individual asset)
            weights[cluster_id] += parent_weight
            return
        
        if cluster_id not in self.cluster_hierarchy:
            return
            
        children = self.cluster_hierarchy[cluster_id]
        n_children = len(children)
        
        if n_children == 0:
            return
        
        # Use Dirichlet distribution to split weight among children
        # Higher alpha = more equal splitting
        alpha = np.ones(n_children) * self.equal_weight_bias
        child_weights = np.random.dirichlet(alpha) * parent_weight
        
        # Recursively assign to children
        for child_id, child_weight in zip(children, child_weights):
            self._recursive_weight_assignment(child_id, child_weight, weights)
    
    def get_name(self) -> str:
        return f"Hierarchical (bias={self.equal_weight_bias:.1f})"

class BayesianOptimizer(SamplingMethod):
    """
    Pure Bayesian optimization directly over portfolio weights.
    Uses Gaussian Process to learn the performance landscape.
    """
    
    def __init__(self, 
                 n_initial_random=25,
                 acquisition_function='ei', 
                 kernel_type='matern52',
                 alpha=1e-3):
        """
        Initialize Bayesian Optimizer
        
        Args:
            n_initial_random: Number of random samples before Bayesian optimization
            acquisition_function: 'ei' (Expected Improvement), 'ucb' (Upper Confidence Bound)
            kernel_type: 'matern52', 'matern32', 'rbf'
            alpha: GP noise parameter (higher = more noise tolerance)
        """
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
        
        # Initialize GP on first call
        if not self.gp_initialized:
            self._initialize_gp()
            self.gp_initialized = True
        
        # Phase 1: Random exploration
        if iteration < self.n_initial_random:
            return self._sample_random_weights(n_assets)
        
        # Phase 2: Bayesian optimization
        if len(self.X_observed) >= 3:
            try:
                return self._bayesian_optimize(n_assets)
            except Exception as e:
                if iteration % 100 == 0:
                    print(f"Bayesian optimization failed (iter {iteration}): {e}")
                return self._sample_random_weights(n_assets)
        else:
            return self._sample_random_weights(n_assets)
    
    def update_observations(self, weights: np.ndarray, performance: float):
        """Update Bayesian optimizer with new observation"""
        self.X_observed.append(weights.copy())
        self.y_observed.append(performance)
        
        # Fit GP with recent observations
        if len(self.X_observed) >= 3 and self.gp is not None:
            try:
                X = np.array(self.X_observed)
                y = np.array(self.y_observed)
                
                # Keep only recent observations to manage complexity
                max_samples = 100
                if len(X) > max_samples:
                    X = X[-max_samples:]
                    y = y[-max_samples:]
                
                self.gp.fit(X, y)
            except Exception as e:
                if self.iteration_count % 100 == 0:
                    print(f"GP fitting failed: {e}")
    
    def _sample_random_weights(self, n_assets):
        """Sample random portfolio weights using Dirichlet distribution"""
        weights = np.random.dirichlet(np.ones(n_assets))
        return weights
    
    def _initialize_gp(self):
        """Initialize Gaussian Process with selected kernel"""
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
        """Optimize portfolio using Bayesian optimization"""
        from scipy.optimize import minimize
        
        def objective(weights):
            # Ensure weights are valid (positive and sum to 1)
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
            return -self._acquisition_function_value(weights.reshape(1, -1))
        
        best_weights = None
        best_value = np.inf
        
        # Try multiple starting points
        starting_points = []
        
        # Start from best observed point
        if len(self.y_observed) > 0:
            best_idx = np.argmax(self.y_observed)
            starting_points.append(self.X_observed[best_idx])
        
        # Add random Dirichlet samples
        for _ in range(5):
            starting_points.append(self._sample_random_weights(n_assets))
        
        # Optimize from each starting point
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
        """Compute acquisition function value"""
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
        """Expected Improvement acquisition function"""
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
        """Approximate normal CDF"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    @staticmethod  
    def _normal_pdf(x):
        """Normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def get_name(self) -> str:
        return f"Bayesian Optimizer ({self.kernel_type.upper()}, {self.acquisition_function.upper()})"


# Export all classes
__all__ = [
    'RandomUniformSampling',
    'RandomSparseSampling', 
    'FastHierarchicalSampling',
    'BayesianOptimizer',
    'FlowHierarchicalSampling'
]