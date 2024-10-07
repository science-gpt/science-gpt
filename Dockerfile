FROM python:3.10-slim

WORKDIR /usr/src/

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /usr/src/app/
COPY ./app .

VOLUME ./vectorstore

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]