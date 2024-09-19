FROM python:3.9-slim

WORKDIR /app

COPY ./requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]