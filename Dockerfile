
FROM python:3.12-slim
LABEL huggingface.co/space-sdk="docker"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY model.joblib .
COPY columns.joblib .
COPY input_data.csv .
RUN ls -la /app
RUN python --version && pip list
EXPOSE 7860
CMD ["waitress-serve", "--host=0.0.0.0", "--port=7860", "--threads=4", "app:app"]
