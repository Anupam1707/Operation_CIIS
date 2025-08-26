# Operation CIIS: Anti-Indian Content Detection

This project is a pipeline for detecting anti-Indian content in social media posts. It includes tools for data scraping, data generation, text cleaning, model training, and inference.

## Features

- **Data Scraping**: Fetches posts from the X API using specific keywords.
- **Synthetic Data Generation**: Creates a large dataset of synthetic posts for training.
- **NLP Model Training**: Fine-tunes a multilingual model (XLM-RoBERTa) to classify content.
- **Inference**: Provides a script to run the trained model on new text.
- **Colab Notebook**: An all-in-one notebook to run the entire pipeline in a Google Colab environment.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Operation_CIIS
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a `.env` file in the root directory and add your X API bearer token:
    ```
    BEARER_TOKEN=your_bearer_token_here
    ```

## Pipeline Workflow

You can run the pipeline using the provided scripts or the Colab notebook.

### Using the Colab Notebook

The easiest way to get started is to use the `Operation_CIIS_Pipeline.ipynb` notebook in Google Colab. It will guide you through all the steps, from installing dependencies to training the model and running inference.

### Running the Scripts Manually

1.  **Generate Synthetic Data:**
    This will create `data/synthetic_posts.json` containing 10,000 synthetic posts.
    ```bash
    python src/data_generation.py
    ```

2.  **Scrape Real Data (Optional):**
    If you have API access, you can scrape real posts. The scraper will use the keywords in your `keywords.json` file and save the posts to your MongoDB database.
    ```bash
    python src/scrape.py
    ```

3.  **Clean the Data:**
    If you have scraped real data, you can clean it using this script. It will process `data/raw_posts.json` and save the cleaned data to `data/cleaned_data.csv` and `data/cleaned_data.json`.
    ```bash
    python src/clean.py --input_file data/your_raw_data.json
    ```

4.  **Train the Model:**
    This script will train the model using the synthetic data (and optionally, the cleaned real data). The trained model will be saved in the `models/multilingual_detector` directory.
    ```bash
    python src/nlp_train.py
    ```

5.  **Run Inference:**
    Use the trained model to classify new text.
    ```bash
    python src/nlp_predict.py --text "Your text to classify here."
    ```
