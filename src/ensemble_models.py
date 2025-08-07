"""
ensemble methods for political bias classification
combining linear svm and logistic regression for optimal performance
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """load the preprocessed training and testing data"""
    print("loading preprocessed data...")
    X_train = joblib.load('processed/X_train.pkl')
    X_test = joblib.load('processed/X_test.pkl')
    y_train = joblib.load('processed/y_train.pkl')
    y_test = joblib.load('processed/y_test.pkl')
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def create_base_models():
    """create base models for ensemble methods"""
    return {
        'lr': LogisticRegression(max_iter=1000, random_state=42),
        'svm': LinearSVC(random_state=42, max_iter=1000),
        'svm_prob': LinearSVC(random_state=42, max_iter=1000)  # for stacking
    }

def create_ensemble_models():
    """create various ensemble models combining lr and svm"""
    
    base_models = create_base_models()
    
    ensembles = {}
    
    # hard voting classifier - uses majority vote from predictions
    ensembles['Hard Voting'] = VotingClassifier(
        estimators=[
            ('lr', base_models['lr']),
            ('svm', base_models['svm'])
        ],
        voting='hard'
    )
    
    # soft voting classifier - uses probabilities
    # note: linearsvc doesn't support predict_proba, using calibrated svm
    from sklearn.calibration import CalibratedClassifierCV
    calibrated_svm = CalibratedClassifierCV(base_models['svm'], cv=3)
    
    ensembles['Soft Voting'] = VotingClassifier(
        estimators=[
            ('lr', base_models['lr']),
            ('svm_cal', calibrated_svm)
        ],
        voting='soft'
    )
    
    # stacking classifier - uses meta-learner to combine predictions
    ensembles['Stacking (LR Meta)'] = StackingClassifier(
        estimators=[
            ('lr', base_models['lr']),
            ('svm', base_models['svm'])
        ],
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    
    # stacking with decision tree meta-learner
    ensembles['Stacking (DT Meta)'] = StackingClassifier(
        estimators=[
            ('lr', base_models['lr']),
            ('svm', base_models['svm'])
        ],
        final_estimator=DecisionTreeClassifier(random_state=42, max_depth=3),
        cv=5
    )
    
    # weighted average ensemble - implemented separately with custom logic
    
    return ensembles

class WeightedEnsemble:
    """custom weighted ensemble combining lr and svm predictions"""
    
    def __init__(self, lr_weight=0.6, svm_weight=0.4):
        self.lr_weight = lr_weight
        self.svm_weight = svm_weight
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.svm_model = LinearSVC(random_state=42, max_iter=1000)
        
    def fit(self, X, y):
        """train both models"""
        self.lr_model.fit(X, y)
        self.svm_model.fit(X, y)
        return self
    
    def predict(self, X):
        """make weighted predictions"""
        # get probability predictions from lr
        lr_probs = self.lr_model.predict_proba(X)
        
        # get decision function scores from svm and convert to probabilities
        svm_scores = self.svm_model.decision_function(X)
        if len(svm_scores.shape) == 1:  # binary case
            svm_probs = np.column_stack([1-svm_scores, svm_scores])
        else:  # multi-class case
            from scipy.special import softmax
            svm_probs = softmax(svm_scores, axis=1)
        
        # weighted combination
        combined_probs = (self.lr_weight * lr_probs + 
                         self.svm_weight * svm_probs)
        
        # return class with highest probability
        return self.lr_model.classes_[np.argmax(combined_probs, axis=1)]
    
    def predict_proba(self, X):
        """Return weighted probabilities"""
        lr_probs = self.lr_model.predict_proba(X)
        
        svm_scores = self.svm_model.decision_function(X)
        if len(svm_scores.shape) == 1:
            svm_probs = np.column_stack([1 - svm_scores, svm_scores])
        else:
            from scipy.special import softmax
            svm_probs = softmax(svm_scores, axis=1)
            
        combined_probs = (self.lr_weight * lr_probs + self.svm_weight * svm_probs)
        return combined_probs

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """evaluate a single model and return results"""
    print(f"\n{'='*50}")
    print(f"evaluating {model_name}")
    print(f"{'='*50}")
    
    # train model
    model.fit(X_train, y_train)
    
    # make predictions
    y_pred = model.predict(X_test)
    
    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"accuracy:  {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"f1-score:  {f1:.4f}")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred
    }

def compare_all_models(X_train, X_test, y_train, y_test):
    """compare individual models and ensemble methods"""
    
    results = {}
    
    # individual models
    print("evaluating individual models")
    individual_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(random_state=42, max_iter=1000)
    }
    
    for name, model in individual_models.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    
    # ensemble models
    print("\nevaluating ensemble models")
    ensemble_models = create_ensemble_models()
    
    for name, model in ensemble_models.items():
        try:
            results[name] = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        except Exception as e:
            print(f"Error with {name}: {e}")
            continue
    
    # custom weighted ensemble
    print("\nevaluating weighted ensemble")
    weighted_ensemble = WeightedEnsemble(lr_weight=0.6, svm_weight=0.4)
    results['Weighted Ensemble'] = evaluate_model(
        weighted_ensemble, X_train, X_test, y_train, y_test, 'Weighted Ensemble'
    )
    
    return results

def perform_cross_validation_ensemble(X_train, y_train):
    """perform cross-validation for ensemble methods"""
    print(f"\n{'='*60}")
    print("cross-validation for ensemble methods")
    print(f"{'='*60}")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(random_state=42, max_iter=1000),
        'Hard Voting': VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000, random_state=42)),
                ('svm', LinearSVC(random_state=42, max_iter=1000))
            ],
            voting='hard'
        )
    }
    
    # add stacking classifier
    models['Stacking'] = StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('svm', LinearSVC(random_state=42, max_iter=1000))
        ],
        final_estimator=LogisticRegression(random_state=42),
        cv=3  # reduced for speed
    )
    
    cv_results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_results[name] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores
            }
            print(f"{name:20} | cv accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        except Exception as e:
            print(f"{name:20} | error: {e}")
    
    return cv_results

def create_summary_report_ensemble(results):
    """create comprehensive summary report for all models"""
    print(f"\n{'='*80}")
    print("comprehensive ensemble model comparison summary")
    print(f"{'='*80}")
    
    # create results dataframe
    df_results = pd.DataFrame({
        name: {
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1_score']:.4f}"
        } for name, result in results.items()
    }).T
    
    print("\nperformance comparison:")
    print("-" * 60)
    print(df_results)
    
    # find best performing models
    best_accuracy = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_f1 = max(results.keys(), key=lambda x: results[x]['f1_score'])
    
    print(f"\ntop performers:")
    print("-" * 60)
    print(f"best accuracy:  {best_accuracy} ({results[best_accuracy]['accuracy']:.4f})")
    print(f"best f1-score:  {best_f1} ({results[best_f1]['f1_score']:.4f})")
    
    # performance improvements
    if 'Logistic Regression' in results and 'Linear SVM' in results:
        lr_f1 = results['Logistic Regression']['f1_score']
        svm_f1 = results['Linear SVM']['f1_score']
        best_ensemble_f1 = max([r['f1_score'] for name, r in results.items() 
                               if name not in ['Logistic Regression', 'Linear SVM']])
        
        improvement_over_lr = ((best_ensemble_f1 - lr_f1) / lr_f1) * 100
        improvement_over_svm = ((best_ensemble_f1 - svm_f1) / svm_f1) * 100
        
        print(f"\nensemble improvements:")
        print("-" * 60)
        print(f"best ensemble f1: {best_ensemble_f1:.4f}")
        print(f"improvement over lr: {improvement_over_lr:+.2f}%")
        print(f"improvement over svm: {improvement_over_svm:+.2f}%")

def save_best_ensemble_model(results):
    """save the best performing ensemble model"""
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]['model']
    
    filename = f"models/best_ensemble_model.pkl"
    joblib.dump(best_model, filename)
    
    print(f"\nsaved best model:")
    print("-" * 60)
    print(f"model: {best_model_name}")
    print(f"f1-score: {results[best_model_name]['f1_score']:.4f}")
    print(f"saved to: {filename}")

def main():
    """main function to run ensemble comparison"""
    print("ensemble learning for political bias classification")
    print("combining linear svm and logistic regression for optimal performance")
    
    # load data
    X_train, X_test, y_train, y_test = load_data()
    
    # compare all models
    results = compare_all_models(X_train, X_test, y_train, y_test)
    
    # cross-validation
    cv_results = perform_cross_validation_ensemble(X_train, y_train)
    
    # create comprehensive report
    create_summary_report_ensemble(results)
    
    # save best model
    save_best_ensemble_model(results)
    
    print(f"\nensemble analysis complete")
    print("key takeaways:")
    print("   - ensemble methods often outperform individual models")
    print("   - voting classifiers combine predictions democratically")
    print("   - stacking uses meta-learning for optimal combination")
    print("   - weighted ensembles allow fine-tuned control")

if __name__ == "__main__":
    main()