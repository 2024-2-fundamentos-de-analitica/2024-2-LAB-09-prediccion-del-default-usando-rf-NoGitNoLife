import pandas as pd
import numpy as np
import os
import json
import gzip
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

class DataProcessor:
    @staticmethod
    def read_data(filepath: str) -> pd.DataFrame:
        """Load dataset from zip file"""
        return pd.read_csv(filepath, index_col=False, compression='zip')
    
    @staticmethod
    def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataset"""
        processed = data.copy()
        processed = (processed
            .rename(columns={'default payment next month': 'default'})
            .drop(columns=['ID'])
            .query("MARRIAGE != 0 and EDUCATION != 0"))
        processed.loc[processed["EDUCATION"] >= 4, "EDUCATION"] = 4
        return processed

class ModelBuilder:
    def __init__(self):
        self.categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
        
    def build_pipeline(self) -> Pipeline:
        """Create preprocessing and model pipeline"""
        preprocessor = ColumnTransformer([
            ('categorical', OneHotEncoder(handle_unknown='ignore'), 
             self.categorical_features)
        ], remainder='passthrough')
        
        model = RandomForestClassifier(random_state=42)
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    
    def create_grid_search(self, pipeline: Pipeline) -> GridSearchCV:
        """Configure grid search with hyperparameters"""
        hyperparameters = {

        'classifier__n_estimators': [50, 100, 200],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_depth': [None, 5, 10, 20],
	    'classifier__min_samples_split': [2, 5, 10],
        }
        
        return GridSearchCV(
            estimator=pipeline,
            cv=10,
            param_grid=hyperparameters,
            n_jobs=-1,
            verbose=2,            
            scoring='balanced_accuracy',
            refit=True
        )

class ModelEvaluator:
    @staticmethod
    def get_performance_metrics(dataset_name: str, y_true, y_pred) -> dict:
        """Calculate precision-based performance metrics"""
        return {
            'type': 'metrics',
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'dataset': dataset_name,
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0))
        }
    
    @staticmethod
    def get_confusion_matrix(dataset_name: str, y_true, y_pred) -> dict:
        """Generate confusion matrix metrics"""
        cm = confusion_matrix(y_true, y_pred)
        return {
            'type': 'cm_matrix',
            'dataset': dataset_name,
            'true_0': {
                "predicted_0": int(cm[0,0]),
                "predicted_1": int(cm[0,1])
            },
            'true_1': {
                "predicted_0": int(cm[1,0]),
                "predicted_1": int(cm[1,1])
            }
        }

class ModelPersistence:
    @staticmethod
    def save_model(filepath: str, model: GridSearchCV):
        """Save model to compressed pickle file"""
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    @staticmethod
    def save_metrics(filepath: str, metrics: list):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            for metric in metrics:
                f.write(json.dumps(metric) + '\n')

def main():
    # Setup paths
    input_path = 'files/input'
    model_path = 'files/models'
    output_path = 'files/output'
    
    # Initialize components
    processor = DataProcessor()
    builder = ModelBuilder()
    evaluator = ModelEvaluator()
    
    # Load and preprocess data
    train_df = processor.preprocess_data(
        processor.read_data(os.path.join(input_path, 'train_data.csv.zip'))
    )
    test_df = processor.preprocess_data(
        processor.read_data(os.path.join(input_path, 'test_data.csv.zip'))
    )
    
    # Split features and target
    X_train = train_df.drop(columns=['default'])
    y_train = train_df['default']
    X_test = test_df.drop(columns=['default'])
    y_test = test_df['default']
    
    # Build and train model
    pipeline = builder.build_pipeline()
    model = builder.create_grid_search(pipeline)
    model.fit(X_train, y_train)
    
    # Save trained model
    ModelPersistence.save_model(
        os.path.join(model_path, 'model.pkl.gz'),
        model
    )
    
    # Generate predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    metrics = [
        evaluator.get_performance_metrics('train', y_train, train_preds),
        evaluator.get_performance_metrics('test', y_test, test_preds),
        evaluator.get_confusion_matrix('train', y_train, train_preds),
        evaluator.get_confusion_matrix('test', y_test, test_preds)
    ]
    
    # Save metrics
    ModelPersistence.save_metrics(
        os.path.join(output_path, 'metrics.json'),
        metrics
    )

if __name__ == "__main__":
    main()