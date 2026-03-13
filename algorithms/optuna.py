from algorithms.algorithm import Algorithm
from abc import abstractmethod
import optuna
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd

class OptunaAlgorithm(Algorithm):
    """
    Abstract base class that integrates Optuna hyperparameter optimization with the asteroid clustering pipeline. 
    It defines a standardized fitness function to evaluate how well a given clustering model recovers 
    known asteroid families, balancing completeness against purity.
    """
    def __init__(self, reload_raw_data=False, algorithm_name="Optuna Base", debug_prints=True, n_trials=100, n_jobs = -1):
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.direction = "maximize"

        super().__init__(reload_raw_data, algorithm_name, debug_prints)

    @abstractmethod
    def define_hyperparams(self, trial: optuna.Trial) -> None:
        pass

    @abstractmethod
    def train_predict(self, params: dict, X_scaled: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_initial_params(self) -> dict | None:
        pass

    def _scale_data(self, params: dict, X: pd.DataFrame) -> np.ndarray:
        # Scaling is dynamically injected into the hyperparameter pipeline because 
        # distance-based density algorithms (like DBSCAN) are highly sensitive to the 
        # geometric distortions introduced by different normalization techniques. 
        # Standard vs. MinMax can drastically alter the density topology of the asteroid proper elements.
        scaler_type = params.get("scaler", "standard") 
        
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "robust":
            lower_q = params.get("robust_lower_q", 25.0)
            upper_q = params.get("robust_upper_q", 75.0)
            scaler = RobustScaler(quantile_range=(lower_q, upper_q))
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
            
        return scaler.fit_transform(X)

    def _get_metrics(self, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Isolates the core metric calculations to prevent redundant contingency matrix 
        # operations between the optimization objective and the final evaluation plots.
        Y_eval = self.Y.fillna("ZZZ_Background").astype(str)
        c_matrix = np.array(contingency_matrix(Y_eval, predictions))
        row_idx, col_idx = linear_sum_assignment(-c_matrix)

        optimal_matches = c_matrix[row_idx, col_idx]
        family_totals   = c_matrix.sum(axis=1)[row_idx]
        cluster_totals  = c_matrix.sum(axis=0)[col_idx]

        with np.errstate(divide='ignore', invalid='ignore'):
            completeness_ratios = optimal_matches / family_totals
            purity_ratios       = optimal_matches / cluster_totals

        # We mask out mappings involving the background noise class and the algorithm's 
        # unclustered noise bin because our fitness function only evaluates discrete, 1-to-1 family recovery.
        valid_mask = (row_idx != c_matrix.shape[0] - 1) & (col_idx != 0)
        valid_comp = completeness_ratios[valid_mask]
        valid_pur  = purity_ratios[valid_mask]
        
        return valid_comp, valid_pur

    def score_predictions(self, predictions: np.ndarray, params: dict) -> float:
        valid_comp, valid_pur = self._get_metrics(predictions)

        # The fitness function heavily penalizes low purity by cubing it. 
        # In asteroid taxonomy, falsely merging distinct, unassociated families into one cluster 
        # is a far worse scientific failure state than artificially fragmenting a single family into subgroups.
        weighted = valid_comp * (valid_pur ** 3.0) 
        
        # A hard floor filter prevents completely failed clusters from generating micro-scores 
        # that could cumulatively drag the optimizer toward a high-quantity, low-quality clustering state.
        score = float(np.sum(weighted[weighted > 0.15]))
        
        # A flat bonus artificially creates a steep gradient in the objective function space, 
        # aggressively pulling the optimizer toward the specific success thresholds we actually care about.
        score += np.sum((valid_comp >= 0.95) & (valid_pur >= 0.80)) * 5.0

        if score > 0:
            successful_95_80 = np.count_nonzero((valid_comp >= 0.95) & (valid_pur >= 0.80))
            pairs = sorted(zip(valid_comp, valid_pur), reverse=True)[:5]
            pairs_str = " | ".join([f"({c:.2f}/{p:.2f})" for c, p in pairs])

        return score

    def objective(self, trial: optuna.Trial) -> float:
        scaler_type = trial.suggest_categorical("scaler", ["standard", "minmax", "robust"])
        if scaler_type == "robust":
            trial.suggest_float("robust_lower_q", 5.0, 25.0)
            trial.suggest_float("robust_upper_q", 75.0, 95.0)

        self.define_hyperparams(trial)
        
        all_params = trial.params

        X_scaled = self._scale_data(all_params, self.X)

        predictions = self.train_predict(all_params, X_scaled)
        
        score = self.score_predictions(predictions, all_params)

        # We inject the raw metric averages into the trial metadata so we can track 
        # the physical trajectory of the optimization (Completeness vs. Purity) 
        # across phase space, instead of relying solely on the abstract scalar fitness score.
        valid_comp, valid_pur = self._get_metrics(predictions)
        mean_comp = float(np.mean(valid_comp)) if len(valid_comp) > 0 else 0.0
        mean_pur = float(np.mean(valid_pur)) if len(valid_pur) > 0 else 0.0
        
        trial.set_user_attr("mean_comp", mean_comp)
        trial.set_user_attr("mean_pur", mean_pur)

        return score

    def _save_evaluation_plot(self, valid_comp: np.ndarray, valid_pur: np.ndarray):
        # Visualizes the final model's performance against the rigid success boundaries. 
        # The dashed lines represent the minimum acceptable thresholds for a predicted 
        # cluster to be classified as a successful family recovery.
        os.makedirs("figures", exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(valid_comp, valid_pur, alpha=0.7, color='#1f77b4', edgecolor='k')
        
        plt.axvline(x=0.95, color='r', linestyle='--', alpha=0.5, label='95% Completeness')
        plt.axhline(y=0.80, color='g', linestyle='--', alpha=0.5, label='80% Purity')
        
        plt.title(f"{self.algorithm_name} - Final Best Model Cluster Performance")
        plt.xlabel("Completeness")
        plt.ylabel("Purity")
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        plot_path = f"figures/{self.algorithm_name}_comp_vs_pur.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Final evaluation plot saved to {plot_path}")

    def _save_optimization_path_plot(self, study: optuna.Study):
        # Maps the temporal evolution of the optimization process. 
        # The red path traces the progression of the "best model so far", revealing 
        # whether the optimizer is successfully climbing the purity gradient 
        # or if it is getting stuck in local optima.
        os.makedirs("figures", exist_ok=True)
        
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials: return

        trial_nums = [t.number for t in trials]
        comps = [t.user_attrs.get("mean_comp", 0.0) for t in trials]
        purs = [t.user_attrs.get("mean_pur", 0.0) for t in trials]
        scores = [t.value for t in trials]

        plt.figure(figsize=(9, 7))
        
        scatter = plt.scatter(comps, purs, c=trial_nums, cmap='viridis', alpha=0.6, edgecolor='k')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Trial Number (Time)')

        best_comp_path, best_pur_path = [], []
        current_best_score = -np.inf
        
        for t, comp, pur, score in zip(trials, comps, purs, scores):
            if score is not None and score > current_best_score:
                current_best_score = score
                best_comp_path.append(comp)
                best_pur_path.append(pur)

        if best_comp_path:
            plt.plot(best_comp_path, best_pur_path, color='red', marker='X', markersize=8, 
                     linewidth=2, label='Path of Best Models', alpha=0.8)

        plt.title(f"{self.algorithm_name} - Optimization Path over Time (Mean Metrics)")
        plt.xlabel("Mean Completeness")
        plt.ylabel("Mean Purity")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        plot_path = f"figures/{self.algorithm_name}_optimization_path.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Optimization path plot saved to {plot_path}")

    def _save_hyperparam_history_plot(self, study: optuna.Study):
        # Diagnoses hyperparameter convergence. If the best trial (red line) sits 
        # at the extreme edge of a plotted parameter boundary, it serves as a visual 
        # indicator that the search space for that specific parameter needs to be expanded.
        os.makedirs("figures", exist_ok=True)
        
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials or len(study.best_params) == 0: return

        param_keys = list(study.best_params.keys())
        num_params = len(param_keys)
        
        fig, axes = plt.subplots(num_params, 1, figsize=(10, 3 * num_params), sharex=True)
        if num_params == 1: axes = [axes] 

        trial_nums = [t.number for t in trials]

        for idx, key in enumerate(param_keys):
            param_vals = [t.params.get(key, np.nan) for t in trials]
            
            axes[idx].scatter(trial_nums, param_vals, color='blue', alpha=0.6, s=20)
            axes[idx].plot(trial_nums, param_vals, color='gray', alpha=0.3, linewidth=1)
            
            axes[idx].axvline(x=study.best_trial.number, color='red', linestyle='--', 
                              alpha=0.7, label=f'Best Trial ({study.best_trial.number})')
            
            axes[idx].set_ylabel(key)
            axes[idx].grid(True, linestyle=':', alpha=0.6)
            axes[idx].legend()

        axes[-1].set_xlabel("Trial Number")
        fig.suptitle(f"{self.algorithm_name} - Hyperparameter Search History", fontsize=14)
        plt.tight_layout()
        
        plot_path = f"figures/{self.algorithm_name}_hyperparams_history.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Hyperparameter history plot saved to {plot_path}")

    def fit_predict(self) -> np.ndarray:
        study_dir = "saved_obj/optuna_studies"
        os.makedirs(study_dir, exist_ok=True)
        study_path = os.path.join(study_dir, f"{self.algorithm_name}_study.pkl")

        # CACHE CHECK LOGIC
        if os.path.exists(study_path):
            if getattr(self, "debug_prints", True):
                print(f"\n[CACHE HIT] Found existing Optuna study for {self.algorithm_name}.")
                print(f"Loading study from {study_path} and bypassing optimization...")
            
            with open(study_path, "rb") as f:
                study = pickle.load(f)
                
            print("\n" + "="*70)
            print("LOADED FROM CACHE")
            print(f"Best Score: {study.best_value}")
            print("Best Parameters:")
            for key, value in study.best_params.items():
                print(f"  - {key}: {value}")
            print("="*70 + "\n")

            # Reconstruct the predictions using the loaded optimal parameters
            X_scaled = self._scale_data(study.best_params, self.X)
            self.cached_predictions = self.train_predict(study.best_params, X_scaled)
            
            # Regenerate the evaluation plot just in case it was deleted or lost
            valid_comp, valid_pur = self._get_metrics(self.cached_predictions)
            self._save_evaluation_plot(valid_comp, valid_pur)

            self._print_top_families(self.cached_predictions, top_n = 8)
            
            return self.cached_predictions

        print(f"\nLAUNCHING OPTUNA OPTIMIZATION FOR {self.algorithm_name}...")
        
        study = optuna.create_study(direction=self.direction)
        
        initial_params = self.get_initial_params()
        if initial_params:
            study.enqueue_trial(initial_params)

        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print(f"Best Score: {study.best_value}")
        print("Best Parameters:")
        for key, value in study.best_params.items():
            print(f"  - {key}: {value}")
        print("="*70 + "\n")

        with open(study_path, "wb") as f:
            pickle.dump(study, f)
            
        print(f"Optuna study saved to {study_path}")

        self._save_optimization_path_plot(study)
        self._save_hyperparam_history_plot(study)

        # We cache the final predictions evaluated on the optimal hyperparameter set 
        # to prevent redundant distance matrix recalculations when external modules 
        # request the clustering results for evaluation metrics.
        X_scaled = self._scale_data(study.best_params, self.X)
        self.cached_predictions = self.train_predict(study.best_params, X_scaled)
        
        valid_comp, valid_pur = self._get_metrics(self.cached_predictions)
        self._save_evaluation_plot(valid_comp, valid_pur)
        
        return self.cached_predictions

    def _print_top_families(self, predictions: np.ndarray, top_n: int = 8):
        """
        Extracts and prints the highest-scoring asteroid families based on the 
        predictions, mapping the contingency matrix rows back to their string labels.
        """
        Y_eval = self.Y.fillna("ZZZ_Background").astype(str)
        family_labels = np.unique(Y_eval) 
        
        c_matrix = np.array(contingency_matrix(Y_eval, predictions))
        row_idx, col_idx = linear_sum_assignment(-c_matrix)

        optimal_matches = c_matrix[row_idx, col_idx]
        family_totals   = c_matrix.sum(axis=1)[row_idx]
        cluster_totals  = c_matrix.sum(axis=0)[col_idx]

        with np.errstate(divide='ignore', invalid='ignore'):
            completeness_ratios = optimal_matches / family_totals
            purity_ratios       = optimal_matches / cluster_totals

        family_stats = []
        for i, (r, c) in enumerate(zip(row_idx, col_idx)):
            label = family_labels[r]
            
            # Skip the background noise class and the algorithm's unclustered noise bin
            if label == "ZZZ_Background" or c == 0:
                continue
                
            comp = completeness_ratios[i]
            pur = purity_ratios[i]
            matches = optimal_matches[i]
            
            # Rank using the same penalty logic as the fitness function
            score = comp * (pur ** 3.0) 
            
            if score > 0:
                family_stats.append({
                    "name": label,
                    "matches": matches,
                    "comp": comp,
                    "pur": pur,
                    "score": score
                })
                
        # Sort descending by score
        family_stats.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"\n--- TOP {top_n} RECOVERED FAMILIES ---")
        print(f"{'Family Name':<20} | {'Matches':<8} | {'Completeness':<12} | {'Purity':<8}")
        print("-" * 55)
        for stat in family_stats[:top_n]:
            # Formatting as percentages for easier reading
            print(f"{stat['name']:<20} | {stat['matches']:<8} | {stat['comp']:<12.2%} | {stat['pur']:<8.2%}")
        print("-" * 55 + "\n")