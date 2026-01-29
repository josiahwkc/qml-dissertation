from sklearn.model_selection import GridSearchCV
from sklearn import svm

class GridSearch():
    param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
    }
    
    def run(self, X_data, y_data):
        grid_search = GridSearchCV(svm.SVC(), self.param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_data, y_data)
        print(f"Best Parameters: {grid_search.best_params_}")

        return grid_search.best_params_['C'], grid_search.best_params_['gamma']