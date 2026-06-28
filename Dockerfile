FROM pythonproject-base:latest

WORKDIR /app
COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]