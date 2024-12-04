FROM python:3.10-slim

WORKDIR /usr/src/

COPY ./requirements.txt .
ARG UPDATE_DEPS=false
RUN if [ "$UPDATE_DEPS" = "true" ]; then \
        pip install --upgrade -r requirements.txt; \
    else \
        pip install -r requirements.txt; \
    fi

WORKDIR /usr/src/data/
COPY ./app/data .

WORKDIR /usr/src/app/
COPY ./app .

VOLUME ./vectorstore

EXPOSE 8501

CMD ["streamlit", "run", "auth.py"]