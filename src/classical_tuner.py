from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

class ClassicalSVMTuner:
    """Optimizes hyperparameters for the classical RBF baseline."""
    
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf']
    }
    
    @classmethod
    def optimize(cls, X_train, y_train):
        """
        Runs 5-fold cross validation to find the optimal baseline SVM.
        """
        # cv=5 inherently uses StratifiedKFold for classification tasks
        grid_search = GridSearchCV(
            estimator=SVC(), 
            param_grid=cls.param_grid, 
            cv=5, 
            n_jobs=-1,
            scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        print(f"Optimal Baseline Params: {grid_search.best_params_}")

        return grid_search.best_estimator_