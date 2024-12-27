# OCR-of-Bank-Statements

This is a Streamlit application designed to process financial documents like salary slips, profit and loss statements, and transaction histories. The app uses OCR (Optical Character Recognition) technologies to extract text, analyze data, and generate insightful visualizations.

## Features
- Extract text using PaddleOCR and EasyOCR.
- Visualize financial data through Bar Charts and Pie Charts.
- Generate answers to user queries using the Together API.
- Support for processing multiple financial documents simultaneously.
- Comparative analysis of financial data.

## Prerequisites
- Python 3.8 or higher
- AWS credentials for accessing the S3 bucket
- Together API key for AI-based query processing

## Installation
1. Clone this repository:
   ```bash
   git clone <https://github.com/AdwaitSalankar/OCR-of-Bank-Statements.git>

2. Navigate to the project directory:
   ```bash
   cd <OCR-of-Bank-Statements-main>

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py

2. Use the sidebar to select OCR engine, document type, and visualization type.
3. Enter the number of images to process and provide a query to analyze the extracted data.
4. Visualize and analyze the results in the app.

## Project Structure
- app.py: Main application code.
- fonts/: Contains font files used by PaddleOCR.
- requirements.txt: Python dependencies.

## Notes
Ensure your AWS credentials are properly configured to access the S3 bucket.
Replace the Together API key in the code with your own key.
Font files required for PaddleOCR should be placed in the fonts folder.
