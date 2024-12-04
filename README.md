# Science-GPT

## Getting started
Install required libraries
```
pip install -r requirements.txt
```

## Running the app locally
From the science-gpt/app directory, run
```
streamlit run auth.py
```

## Running the full app via docker compose
This is necessary to run with Milvus.
```
./run_science_gpt.sh
```
If the script fails try running with sudo.

To bypass authentication for dev purposes run
```
./run_science_gpt.sh --dev
```

To rebuild the images run
```
./run_science_gpt.sh --build
```

To update dependencies run
```
./run_science_gpt.sh --update-deps
```

The --dev, --build, and --update-deps flags can be used together.
