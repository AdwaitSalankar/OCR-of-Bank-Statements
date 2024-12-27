import boto3
import random
import streamlit as st
import easyocr
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from PIL import Image, ImageDraw
import cv2
from together import Together
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from io import StringIO, BytesIO

# Together API setup
def query_together_api(prompt):
    client = Together(api_key="fe6be49f616e3a5c034bccf07c5d6bc784e33f3eed4ab7919a57659feed544ff") 
    response = client.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        prompt=prompt,
        max_tokens=500,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        stop=["</s>"],
        stream=True
    )
    answer = ""
    for token in response:
        if hasattr(token, 'choices'):
            answer += token.choices[0].delta.content
    return answer.strip()

# Initialize OCR models
easyocr_reader = easyocr.Reader(['en'], gpu=True)
paddleocr_reader = PaddleOCR(use_angle_cls=True)

# Font path for PaddleOCR
font_path = os.path.join("fonts", "DejaVuSans.ttf")
if not os.path.exists(font_path):
    st.warning("Font file not found, using default font.")
    font_path = None  # Use default font if the file is not found

# AWS S3 setup
s3 = boto3.client(
        's3',
        aws_access_key_id='AKIAZDZTBO2VQC2H7ERF',
        aws_secret_access_key='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'
    )

    # Bucket name and folders
bucket_name = 'bankdataset-ocr'
folders = {
        'Profit and Loss Statement': 'profit_and_loss_statement/',
        'Salary Slip': 'salaryslip/',
        'Transaction History': 'transaction_history/'
    }

def generate_query_prompt(extracted_text, user_question):
    return (
        f"You are an intelligent assistant. Use the following text extracted from a document:\n\n"
        f"{extracted_text}\n\n"
        f"Perform the requested task described below:\n"
        f"Task: {user_question}\n"
        f"Do not generate follow-up questions. Provide a single answer focusing on the direct information available in the text."
    )

def extract_fields_prompt(extracted_text):
    if document_type == "Salary Slip":
        return (
            f"Extract only Salary Components values from the following text:\n"
            f"{extracted_text}\n\n"
            f"Only provide the output as a csv format with 'Field' and 'Value' as keys."
            f"Do not provide any additional code or analysis."
            f"When returning field, change the field name to a standard name and then return"
        )
    elif document_type == "Profit and Loss Statement":
        return (
            f"Extract Profit and Loss Statement Components values from the following text:\n"
            f"{extracted_text}\n\n"
            f"Only provide the output as a csv format with 'Field' and 'Value' as keys."
        )
    elif document_type == "Balance Sheet":
        return (
            f"Extract Balance Sheet Components values from the following text:\n"
            f"{extracted_text}\n\n"
            f"Only provide the output as a csv format with 'Field' and 'Value' as keys."
        )

def plot_comparative_fields(all_data):
    """
    Create a comparative visualization of salary fields across multiple images.
    
    Args:
        all_data (list): List of dictionaries containing image names and their salary data
    """
    if chart_type == "Bar Chart":
        # Collect all unique fields
        all_fields = set()
        for data in all_data:
            all_fields.update(data['fields'])

        # Prepare data for plotting
        unique_fields = sorted(list(all_fields))
        
        # Filter to top 5 fields for each image data
        for data in all_data:
            # Sort fields by their values in descending order and keep the top 5
            sorted_fields = sorted(data['values'].items(), key=lambda x: x[1], reverse=True)
            top_fields = sorted_fields[:5]
            # Update the data with the top 5 fields only
            data['fields'] = [field for field, value in top_fields]
            data['values'] = {field: value for field, value in top_fields}

        # Recompute unique fields based on the top fields only
        all_fields = set()
        for data in all_data:
            all_fields.update(data['fields'])

        unique_fields = sorted(list(all_fields))

        # Set up the plot
        plt.figure(figsize=(20, 12))
        
        # Create a grouped bar plot
        x = np.arange(len(unique_fields))
        width = 0.7 / len(all_data)
        
        # Color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_data)))
        
        # Plot bars for each image
        for i, data in enumerate(all_data):
            # Prepare values, filling in 0 for missing fields
            values = [data['values'].get(field, 0) for field in unique_fields]
            
            plt.bar(x + i*width, values, width, 
                    label=data['name'], 
                    color=colors[i], 
                    edgecolor='black', 
                    linewidth=1.5)
            
            # Add value annotations
            for j, v in enumerate(values):
                if v > 0:
                    plt.text(x[j] + i*width, v + 600, f'₹{v:,.0f}', 
                            ha='center', va='bottom', 
                            fontsize=14, rotation=40, color="darkblue")
        
        # Customize the plot
        plt.xlabel("Components", fontsize=14, fontweight="bold")
        plt.ylabel("Amount (₹)", fontsize=14, fontweight="bold")
        plt.title("Comparative Graph (Top 5 Values)", fontsize=14, fontweight="bold", color="darkblue")
        plt.xticks(x + width * (len(all_data) - 1) / 2, unique_fields, rotation=40, ha='right', fontsize=10)

        plt.legend(title="Images", loc='upper left', fontsize=12, title_fontsize=14, frameon=True)

        plt.tight_layout(pad=2)
        
        # Display the plot in Streamlit
        st.pyplot(plt)
        
        # Create a comparative summary table
        summary_data = []
        for data in all_data:
            image_summary = {'Image Name': data['name']}
            image_summary.update({field: f'₹{data["values"].get(field, 0):,.2f}' for field in unique_fields})
            summary_data.append(image_summary)
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df.set_index('Image Name'), use_container_width=True)

    elif chart_type == "Pie Chart":
        if len(all_data) < 2:
            st.error("Please upload at least two images to compare pie charts.")
            return

        # Split the data into groups of two for side-by-side comparison
        grouped_data = [all_data[i:i + 2] for i in range(0, len(all_data), 2)]

        for group in grouped_data:
            # Create a figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))  # Two charts side by side
            colors = plt.cm.Set3(np.linspace(0, 1, 8))  # Limit to 8 colors

            # Ensure axes is iterable (even if there's only one chart in the group)
            if len(group) < 2:
                axes = [axes, None]

            for i, data in enumerate(group):
                if data is not None:
                    # Prepare data for the pie chart
                    sorted_data = sorted(data['values'].items(), key=lambda x: x[1], reverse=True)[:8]  # Top 8 fields
                    labels, sizes = zip(*sorted_data)  # Separate fields and values
                    chart_colors = colors[:len(labels)]  # Adjust colors to match the number of fields

                    # Plot the pie chart
                    wedges, texts, autotexts = axes[i].pie(
                        sizes, labels=None, autopct='%1.1f%%', startangle=140, colors=chart_colors
                    )
                    axes[i].set_title(data['name'], fontsize=12, fontweight="bold")

                    # Add a legend for the current chart
                    axes[i].legend(
                        handles=[
                            plt.Line2D([0], [0], marker='o', color=color, lw=0) for color in chart_colors
                        ],
                        labels=[f"{label}: ₹{value:,.0f}" for label, value in sorted_data],
                        title="Fields & Values",
                        loc='lower center',
                        bbox_to_anchor=(0.5, -0.2),
                        fontsize=10,
                        frameon=False,
                        ncol=2
                    )
                else:
                    # Hide empty subplot
                    fig.delaxes(axes[i])

            # Adjust layout for legends and charts
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.3)  # Adjust to fit the legends
            plt.suptitle("Comparative Pie Charts of Salary Components (Top 8)", fontsize=14, fontweight="bold", color="darkblue")
            st.pyplot(fig)

def process_single_image(image, ocr_option):
    """
    Process a single image and extract salary fields
    
    Args:
        image (PIL.Image): Uploaded image
        ocr_option (str): Selected OCR engine
    
    Returns:
        dict: Extracted text and salary fields
    """
    image_np = np.array(image)

    if len(image_np.shape) == 2:  # Grayscale image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # RGBA image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    extracted_text = ""

    # Perform OCR and extract text
    if ocr_option in ["EasyOCR"]:
        easyocr_result = easyocr_reader.readtext(image_np)
        easyocr_text_only = "\n".join([text for _, text, _ in easyocr_result])
        extracted_text += easyocr_text_only + "\n"

    if ocr_option in ["PaddleOCR"]:
        paddleocr_result = paddleocr_reader.ocr(image_np, cls=True)
        paddleocr_text_only = "\n".join([line[1][0] for line in paddleocr_result[0]])
        extracted_text += paddleocr_text_only + "\n"

    # Extract fields and values
    fields_prompt = extract_fields_prompt(extracted_text)
    fields_response = query_together_api(fields_prompt)

    # Parse the fields and values
    fields = []
    values = {}

    try:
        csv_data = StringIO(fields_response)
        reader = csv.reader(csv_data)
        
        # Skip the header row
        next(reader, None)
        
        for row in reader:
            if row and len(row) >= 2:
                # Clean the field names and values
                field = row[0].strip('" ')
                value_str = row[1].strip('" ').replace('```', '').replace(',', '')
                
                try:
                    # Use float() to handle decimal values
                    value = float(value_str)
                    fields.append(field)
                    values[field] = value
                except ValueError:
                    pass

    except Exception as e:
        st.error(f"An error occurred while processing the salary information: {e}")

    return {
        'text': extracted_text,
        'fields': fields,
        'values': values
    }

def download_random_images_from_s3(document_type, num_images):

    folder = folders.get(document_type, '')
    if not folder:
        st.error("Invalid document type selected.")
        return []

    try:
        # List objects in the specified folder
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder)
        if 'Contents' not in objects:
            st.error("No images found in the specified folder.")
            return []

        # Get a list of image keys
        image_keys = [obj['Key'] for obj in objects['Contents'] if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Select random images
        selected_keys = random.sample(image_keys, min(num_images, len(image_keys)))

        # Download images
        downloaded_images = []
        for key in selected_keys:
            response = s3.get_object(Bucket=bucket_name, Key=key)
            image_data = response['Body'].read()
            image = Image.open(BytesIO(image_data))
            downloaded_images.append(image)

        return downloaded_images

    except boto3.exceptions.Boto3Error as e:
        st.error(f"An error occurred while accessing Database: {e}")
        return []

# Existing code for OCR processing and visualization...

# Streamlit App
st.title("📈 OCR for Financial Insights & Document Analyzer")
st.write("Select number of Images and Enter Prompt to Extract Insights from Documents")
num_images = st.number_input("Enter the number of images to compare", min_value=2, step=1)
# Input query
user_question = st.text_input("Ask a question about the extracted text:", placeholder="e.g.: Summarize it")
st.sidebar.title("Choose Type:")

# OCR Option Selection
ocr_option = st.sidebar.radio("Select OCR Engine:", ( "PaddleOCR", "EasyOCR"), index=0)

# Document type selection
document_type = st.sidebar.radio(
    "Select Document Type:", ["Salary Slip", "Profit and Loss Statement", "Transaction History"]
)

# Add visualization type selection
chart_type = st.sidebar.selectbox(
    "Select Visualization Type:",
    ("Bar Chart", "Pie Chart")
)

# Initialize session state
if 'downloaded_images' not in st.session_state:
    st.session_state.downloaded_images = []
if 'all_image_data' not in st.session_state:
    st.session_state.all_image_data = []
if 'responses' not in st.session_state:
    st.session_state.responses = []

# Function to reset responses when needed
def reset_responses():
    st.session_state.responses = []

# Download and process images from S3
if st.button("Visualize"):
    with st.spinner("Downloading images..."):
        st.session_state.downloaded_images = download_random_images_from_s3(document_type, num_images)

    if st.session_state.downloaded_images:
        st.session_state.all_image_data = []
        st.session_state.responses = []
        count = 1

        for image in st.session_state.downloaded_images:
            with st.spinner(f"Processing image {count}..."):
                image_data = process_single_image(image, ocr_option)
                image_data['name'] = f"Image {count}"
                st.session_state.all_image_data.append(image_data)
                extract_text = image_data['text']
                answer = generate_query_prompt(extract_text, user_question)
                response = query_together_api(answer)
                
                if response:
                    st.session_state.responses.append(f"Answer for image {count}: {response}")
                else:
                    st.session_state.responses.append(f"Answer for image {count}: Could not find a relevant answer. Try rephrasing your query.") 
                
                count += 1

# Display responses
if st.session_state.responses:
    for idx, response in enumerate(st.session_state.responses, start=1):
        st.write(f"Response for image {idx}:")
        if "Could not find a relevant answer" in response:
            st.warning(response)
        else:
            st.success(response)

if 'selected_image_index' not in st.session_state:
    st.session_state.selected_image_index = 0 
    
# Display downloaded images
if st.session_state.downloaded_images:
    with st.expander("Downloaded Images", expanded=True):  # Keep expander open by default
        # Create a mapping of image labels to actual image objects
        image_labels = [f"Image {idx+1}" for idx in range(len(st.session_state.downloaded_images))]

        # Use session state to remember the selected image
        selected_label = st.selectbox(
            "Select an image to view",
            options=image_labels,
            index=st.session_state.selected_image_index,
            key="image_selector"  # Key to track this widget
        )

        # Update session state with the current selection
        st.session_state.selected_image_index = image_labels.index(selected_label)

        # Find the corresponding image object
        selected_image = st.session_state.downloaded_images[st.session_state.selected_image_index]

        # Display the selected image
        st.image(selected_image, caption=selected_label)

        # Comparative Visualization
    if st.session_state.all_image_data:
            st.header(f"Comparative Analysis: {document_type}")
            plot_comparative_fields(st.session_state.all_image_data)

            # Display extracted text for each image
            for idx, image_data in enumerate(st.session_state.all_image_data):
                with st.expander(f"OCR Results: Image {idx + 1}"):
                    st.subheader(f"OCR Results: Image {idx + 1}")
                    lines = image_data['text'].split('\n')
                    df = pd.DataFrame({'Extracted Text': lines})
                    st.dataframe(df, use_container_width=True)
else:
    st.warning("No images downloaded from Database.")     