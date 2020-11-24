import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression

from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_transformer', DistanceTransformer()),
        ('standard_scaler', StandardScaler())])
        
        time_pipe = Pipeline([
        ('time_encoder', TimeFeaturesEncoder('pickup_datetime')),
        ('cat_encoder', OneHotEncoder(handle_unknown='ignore'))])
        
        prepro_pipe = ColumnTransformer([
        ('Distance', dist_pipe, ["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]),
        ('Time', time_pipe, ['pickup_datetime'])], remainder='drop')
        
        model_pipe = Pipeline([
        ('preprocessing', prepro_pipe),
        ('model', LinearRegression())])
        
        self.pipeline = model_pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        print(compute_rmse(self.pipeline.predict(self.X_test), self.y_test))


if __name__ == "__main__":
    # get data and clean data
    df = clean_data(get_data())
    # set X and y
    features = ['key', 'pickup_datetime', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count']
    X, y = df[features], df['fare_amount']
    # train
    trainer = Trainer(X, y)
    trainer.set_pipeline()
    trainer.run()
    # evaluate
    trainer.evaluate()
