from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

class ClassicalSVMTuner:
    """Optimizes hyperparameters ONCE per dataset, reuses across trials"""
    
    _cached_params = {}
    
    @classmethod
    def get_best_params(cls, X_train, y_train, cache_key=None):
        """
        Find best hyperparameters once, cache for reuse.
        
        Args:
            cache_key: Unique identifier for this dataset/config
        """
        if cache_key and cache_key in cls._cached_params:
            print(f"Using cached hyperparameters for {cache_key}")
            return cls._cached_params[cache_key]
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 0.001, 0.01, 0.1, 1]
        }
        
        grid_search = GridSearchCV(
            SVC(kernel='rbf'), 
            param_grid, 
            cv=5, 
            n_jobs=-1,
            scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        
        print(f"Optimal params: {best_params}")
        
        if cache_key:
            cls._cached_params[cache_key] = best_params
        
        return best_params