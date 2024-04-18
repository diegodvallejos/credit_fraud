from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,  cross_val_score
import pickle

class Trainer:
    def __init__(self, data):
        self.data = data

    def split_data(self):
        self.target = self.data['Class']
        self.features = self.data.drop('Class', axis=1)
        
        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(self.features, self.target, test_size=0.2, random_state=42)
        self.train_data, self.val_data, self.train_target, self.val_target = train_test_split(self.train_data, self.train_target, test_size=0.1, random_state=42)
    def train_models(self):
        models = {
            'Logistic Regression': LogisticRegression(),
            'XGBoost Classifier': XGBClassifier(),
            'LightGBM': LGBMClassifier(),
            'Random Forest': RandomForestClassifier()
        }
        
        best_model = None
        best_f1_score = 0
        
        for model_name, model in models.items():
            cv = 5  # Number of folds for cross validation
            
            random_search = RandomizedSearchCV(model, param_distributions={}, scoring='f1', n_iter=10, cv=cv)
            random_search.fit(self.train_data, self.train_target)
            
            if random_search.best_score_ > best_f1_score:
                best_model = random_search.best_estimator_
                best_f1_score = random_search.best_score_
        
        return self.best_model
    
    def save_best_model(self, model):
        with open('best_model.pkl', 'wb') as file:
            pickle.dump(model, file)
