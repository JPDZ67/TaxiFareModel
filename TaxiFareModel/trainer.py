from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV

MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "Josep"
EXPERIMENT_NAME = f"TaxifareModel_{myname}"

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_distance = Pipeline(steps = [ ('distance_transformer', DistanceTransformer()),
                                   ('distance_scaling', StandardScaler()) ] )
        pipe_time = Pipeline([ ('time_transformer', TimeFeaturesEncoder("pickup_datetime")),
                        ('time_encode', OneHotEncoder(handle_unknown='ignore',sparse=False)) ])
        pipe_passengers = Pipeline([ ('passenger_scaler', RobustScaler())])
        
        distance_columns = ["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]
        time_columns     = ['pickup_datetime']
        passenger_columns= ['passenger_count']

        preproc_pipe = ColumnTransformer([('distance', pipe_distance, distance_columns),
                                          ('time', pipe_time, time_columns ), 
                                          ('passenger', pipe_passengers, passenger_columns )], remainder = 'drop')
        
        self.pipeline = Pipeline([ ('transformer', preproc_pipe),('regressor', LassoCV(cv=5, n_alphas=5)) ] )

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse_ = compute_rmse (y_pred, y_test)
        
        print(f"RMSE = {rmse_}")

        self.experiment_name = EXPERIMENT_NAME

        self.mlflow_log_metric("rmse", rmse_)
        self.mlflow_log_param("model", "LassoCV(cv=5, n_alphas=5)")
        self.mlflow_log_param("student_name", myname)

        return rmse_

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    df = get_data(10000)
    # clean data
    df_cleaned = clean_data(df)
    # set X and y
    target = "fare_amount"
    features = list(df.drop(columns= [target]).columns)
    X = df_cleaned[features]; y = df_cleaned[target]
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    # train
    model = Trainer(X_train,y_train)
    model.run()
    # evaluate
    rmse = model.evaluate(X_test,y_test)
    print('TODO ...')
