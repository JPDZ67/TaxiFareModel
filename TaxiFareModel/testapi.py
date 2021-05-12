import requests

taxifare_api_url = f"http://localhost:8080/predict_fare?pickup_datetime=2013-07-06 17:18:00 UTC&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=1"

response = requests.get(
    taxifare_api_url
).json()

print(response)

# if __name__ == '__main__':

