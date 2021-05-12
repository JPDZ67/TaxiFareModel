FROM python:3.8.10-buster
COPY api /api
COPY XGBoost_finalized_model.pkl /XGBoost_finalized_model.pkl
COPY TaxiFareModel /TaxiFareModel
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
CMD uvicorn api.fast:app --host 0.0.0.0