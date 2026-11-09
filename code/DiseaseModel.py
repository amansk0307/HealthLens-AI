import xgboost as xgb
import pandas as pd
from code.disease_ranking import rank_disease_probabilities

class DiseaseModel:

    def __init__(self):
        self.all_symptoms = None
        self.symptoms = None
        self.pred_disease = None
        self.model = xgb.XGBClassifier()
        self.diseases = self.disease_list('data/dataset.csv')

    def load_xgboost(self, model_path):
        self.model.load_model(model_path)

    def save_xgboost(self, model_path):
        self.model.save_model(model_path)

    def predict(self, X):
        prediction = self.predict_top_k(X, top_k=1)[0]
        self.pred_disease = prediction[0]
        return prediction

    def predict_top_k(self, X, top_k=3):
        self.symptoms = X
        probabilities = self.model.predict_proba(self.symptoms)[0]
        class_labels = getattr(self.model, "classes_", range(len(probabilities)))
        predictions = rank_disease_probabilities(
            probabilities,
            class_labels,
            self.diseases,
            top_k=top_k,
        )
        if predictions:
            self.pred_disease = predictions[0][0]
        return predictions

    
    def describe_disease(self, disease_name):

        if disease_name not in self.diseases:
            return "That disease is not contemplated in this model"
        
        # Read disease dataframe
        desc_df = pd.read_csv('data/symptom_Description.csv')
        desc_df = desc_df.apply(lambda col: col.str.strip())
        descriptions = desc_df[desc_df['Disease'] == disease_name]['Description']
        if descriptions.empty:
            return "No dataset description is available for this model label."

        return descriptions.values[0]

    def describe_predicted_disease(self):

        if self.pred_disease is None:
            return "No predicted disease yet"

        return self.describe_disease(self.pred_disease)
    
    def disease_precautions(self, disease_name):

        if disease_name not in self.diseases:
            return "That disease is not contemplated in this model"

        # Read precautions dataframe
        prec_df = pd.read_csv('data/symptom_precaution.csv')
        prec_df = prec_df.apply(lambda col: col.str.strip())

        return prec_df[prec_df['Disease'] == disease_name].filter(regex='Precaution').values.tolist()[0]

    def predicted_disease_precautions(self):

        if self.pred_disease is None:
            return "No predicted disease yet"

        return self.disease_precautions(self.pred_disease)

    def disease_list(self, kaggle_dataset):

        df = pd.read_csv('data/clean_dataset.tsv', sep='\t')
        # Preprocessing
        y_data = df.iloc[:,-1]
        X_data = df.iloc[:,:-1]

        self.all_symptoms = X_data.columns

        # Convert y to categorical values
        y_data = y_data.astype('category')
        
        return y_data.cat.categories