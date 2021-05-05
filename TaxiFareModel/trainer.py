from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV


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
        pipe_passengers = Pipeline([ ('passenger_scaler', StandardScaler())])
        
        distance_columns = ["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]
        time_columns     = ['pickup_datetime']
        passenger_columns= ['passenger_count']

        preproc_pipe = ColumnTransformer([('distance', pipe_distance, distance_columns),
                                          ('time', pipe_time, time_columns ), 
                                          ('passenger', pipe_passengers, passenger_columns )], remainder = 'drop')
        
        pipe_model = Pipeline([ ('transformer', preproc_pipe),('regressor', LassoCV(cv=5, n_alphas=5)) ] )

        self.pipeline = pipe_model

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse_ = compute_rmse (y_pred, y_test)
        print(f"RMSE = {rmse_}")
        return rmse_


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
    print('TODO')
