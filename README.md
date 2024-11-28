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
```
./run_science_gpt.sh
```
If the script fails try running with sudo.

To bypass authentication for dev purposes run
```
./run_science_gpt.sh --dev
```

And to rebuild the images run
```
./run_science_gpt.sh --build
```

The --build and --dev flags can be used together.