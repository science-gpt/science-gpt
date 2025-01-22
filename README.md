# Science-GPT

## Getting Started

Before installing the required libraries, it is recommended to set up a virtual environment. This ensures that the dependencies for this project do not interfere with other Python projects on your system.

### Creating a Virtual Environment

You can create a virtual environment using either `venv` or `conda`:

#### Using `venv`:

1. Navigate to your project directory:
   ```bash
   cd science-gpt
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

#### Using `conda`:

1. Create a new conda environment:
   ```bash
   conda create -n science-gpt python=3.10
   ```
2. Activate the conda environment:
   ```bash
   conda activate science-gpt
   ```

### Install Required Libraries

Once the virtual environment is activated, install the required libraries:

```bash
pip install -r requirements.txt
```

### Configure User Authentication

If you are not logged in before, navigate to the `app/src/configs/user_config.yaml` file. Scroll down to the section:

```yaml
pre-authorized:
  emails:
```

Add your _Health Canada_ _email_ to the list under `emails`.

### Preparing for File Input

This project is a robust Retrieval-Augmented Generation (RAG) system. For temporary use, upload the PDF files you want to feed into the LLM by placing them in the `app/data` folder.

## Running the Full App via Docker Compose

### Ensure Docker is Installed

Before proceeding, make sure Docker is installed on your computer and running. You can download Docker Desktop from [here](https://www.docker.com/products/docker-desktop/).

### Update Configuration

Navigate to `app/src/configs/user_config.yaml`. On line 93, ensure the `vector_db` settings are correct:

```yaml
vector_db:
  database: "milvus"
  host: "localhost"
  port: 19530
  supported_databases:
    - "chromadb"
    - "milvus"
```

Verify that the `host` setting is set to `"localhost"`.

### Running the App

Run the following command from the project root:

```bash
./run_science_gpt.sh
```

If the script fails, try running it with `sudo`:

```bash
sudo ./run_science_gpt.sh
```

### Development Mode

To bypass authentication for development purposes, use the `--dev` flag:

```bash
./run_science_gpt.sh --dev
```

### Rebuilding Docker Images

To rebuild the Docker images, use the `--build` flag:

```bash
./run_science_gpt.sh --build
```

### Updating Dependencies

To update the dependencies, use the `--update-deps` flag:

```bash
./run_science_gpt.sh --update-deps
```

### Combining Flags

You can combine the `--dev`, `--build`, and `--update-deps` flags as needed. For example:

```bash
./run_science_gpt.sh --dev --build --update-deps
```

## Running the App Locally

From the `science-gpt/app` directory, run:

```bash
cd app
```

Then, run the following command:

```bash
streamlit run auth.py
```

If running for the first time, use the Health Canada email added to the `user_config.yaml` file to complete the registration. Your email will be hashed/encrypted into the credentials stored in `user_config.yaml`.

Once you successfully log in, feel free to explore the chatbot. For now, you can select GPT-4.0 and GPT-3.5 to test it and input your queries. 

### Managing Files

To upload or delete files, add or remove them from the `app/data` folder, then rerun:

```bash
streamlit run auth.py
```

