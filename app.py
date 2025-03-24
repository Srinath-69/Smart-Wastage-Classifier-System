from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st 
from dotenv import load_dotenv 
import os
import google.generativeai as genai
import base64

# Set page configuration (must be the first Streamlit command)
st.set_page_config(layout='wide')

# Function to encode image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Custom HTML for the title with logo beside it
title_box_html = """
<div style="display: flex; align-items: center; justify-content: center; padding: 20px; background-color: rgba(255, 255, 255, 0.8); border-radius: 10px;">
    <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="width: 50px; height: 50px; margin-right: 10px;">
    <h1 style="text-align: center; color: black; margin: 0;">Smart Wastage Sorting System</h1>
</div>
"""

# Encode the logo image to Base64
logo_base64 = get_base64_image("static/logo.png")

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Load and encode the PNG image
img_base64 = get_base64_image("static/background.png")  # Replace with your PNG file path

# Custom CSS to set the background image
background_css = f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{img_base64}");
    background-size: cover;  /* Scales the image to cover the entire app */
    background-position: center;  /* Centers the image */
    background-repeat: no-repeat;  /* Prevents tiling */
}}
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(background_css, unsafe_allow_html=True)

# Inject the HTML into the Streamlit app
st.markdown(title_box_html.format(logo_base64=logo_base64), unsafe_allow_html=True)

def classify_waste(img):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=True)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Replace this with the path to your image
    image = img.convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    #print("Class:", class_name[2:], end="")
    #print("Confidence Score:", confidence_score)

    return class_name, confidence_score

generation_config = {
"temperature": 0.7,
"top_p": 1,
"top_k": 1,
"max_output_tokens": 600,
}

safety_settings = [
{
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
},
{
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
},
{
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
},
{
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
},
]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-002",
                            generation_config=generation_config,
                            safety_settings=safety_settings)

def generate_carbon_footprint_info(label):
    label = label.split(' ')[1]
    print(label)
    prompt = f"What is the approximate Carbon emission or carbon footprint generated from {label}? I just need an approximate number to create awareness. Elaborate in 100 words."
    convo = model.start_chat()
    convo.send_message(prompt)
    result_text = convo.last.text
    st.markdown(f"""
    <div style="background-color: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 10px;">
        <p style="text-align: center; color: black;">{result_text}</p>
    </div>
    """, unsafe_allow_html=True)

input_img = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'], label_visibility="hidden")

if input_img is not None:
    if st.button("Classify"):
        
        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">Your Uploaded Image</p>
            </div>
            """, unsafe_allow_html=True)
            st.image(input_img, use_container_width=True)

        with col2:
            st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">Your Result</p>
            </div>
            """, unsafe_allow_html=True)
            image_file = Image.open(input_img)
            label, confidence_score = classify_waste(image_file)
            col4, col5 = st.columns([1,1])
            if label == "0 battery\n":
                st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">The image is classified as BATTERY, Which comes under "HAZARDOUS WASTE OR E-WASTE"</p>
            </div>
            """, unsafe_allow_html=True)                
                with col4:
                    st.image("sdg goals/12.gif", use_container_width=True)
                    st.image("sdg goals/3.gif", use_container_width=True)
                    st.image("sdg goals/14.gif", use_container_width=True)
                with col5:
                    st.image("sdg goals/6.gif", use_container_width=True)
                    st.image("sdg goals/11.gif", use_container_width=True)
                    st.image("sdg goals/15.gif", use_container_width=True) 
            elif label == "1 brown-glass\n":
                st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">The image is classified as BROWN-GLASS, Which comes under "GLASS-RECYCLING"</p>
            </div>
            """, unsafe_allow_html=True)
                with col4:
                    st.image("sdg goals/11.gif", use_container_width=True)
                    st.image("sdg goals/12.gif", use_container_width=True)
                    st.image("sdg goals/6.gif", use_container_width=True)
                with col5:
                    st.image("sdg goals/13.gif", use_container_width=True)
                    st.image("sdg goals/15.gif", use_container_width=True)
                    st.image("sdg goals/9.gif", use_container_width=True) 
            elif label == "2 cardboard\n":
                st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">The image is classified as CARDBOARD, Which comes under "CORRUGATED-CARDBOARD"</p>
            </div>
            """, unsafe_allow_html=True)
                with col4:
                    st.image("sdg goals/11.gif", use_container_width=True)
                    st.image("sdg goals/12.gif", use_container_width=True)
                with col5:
                    st.image("sdg goals/13.gif", use_container_width=True)
                    st.image("sdg goals/15.gif", use_container_width=True)
            elif label == "3 clothes\n":
                st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">The image is classified as CLOTHES, Which comes under "TEXTILE OR FABRIC RECYCLING"</p>
            </div>
            """, unsafe_allow_html=True)
                with col4:
                    st.image("sdg goals/3.gif", use_container_width=True)
                    st.image("sdg goals/6.jpg", use_container_width=True)
                with col5:
                    st.image("sdg goals/12.gif", use_container_width=True)
                    st.image("sdg goals/14.gif", use_container_width=True)
            elif label == "4 green-glass\n":
                st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">The image is classified as GREEN-GLASS, Which comes under "GLASS-RECYCLING"</p>
            </div>
            """, unsafe_allow_html=True)
                with col4:
                    st.image("sdg goals/6.gif", use_container_width=True)
                    st.image("sdg goals/9.gif", use_container_width=True)
                    st.image("sdg goals/11.gif", use_container_width=True)
                with col5:
                    st.image("sdg goals/12.gif", use_container_width=True)
                    st.image("sdg goals/13.gif", use_container_width=True)
                    st.image("sdg goals/15.gif", use_container_width=True)
            elif label == "5 metal\n":
                st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">The image is classified as METAL, Which comes under "METAL-RECYCLING"</p>
            </div>
            """, unsafe_allow_html=True)
                with col4:
                    st.image("sdg goals/6.gif", use_container_width=True)
                    st.image("sdg goals/8.gif", use_container_width=True)
                    st.image("sdg goals/9.gif", use_container_width=True)
                with col5:
                    st.image("sdg goals/11.gif", use_container_width=True)
                    st.image("sdg goals/12.gif", use_container_width=True)
                    st.image("sdg goals/13.gif", use_container_width=True) 
            elif label == "6 organic\n":
                st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">The image is classified as ORGANIC, Which comes under "COMPOSTABLE-WASTE OR ORGANIC RECYCLING"</p>
            </div>
            """, unsafe_allow_html=True)
                with col4:
                    st.image("sdg goals/2.gif", use_container_width=True)
                    st.image("sdg goals/6.gif", use_container_width=True)
                    st.image("sdg goals/7.gif", use_container_width=True)
                with col5:
                    st.image("sdg goals/11.gif", use_container_width=True)
                    st.image("sdg goals/12.gif", use_container_width=True)
                    st.image("sdg goals/13.gif", use_container_width=True)
            elif label == "7 paper\n":
                st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">The image is classified as PAPER, Which comes under "PAPER-RECYCLING"</p>
            </div>
            """, unsafe_allow_html=True)
                with col4:
                    st.image("sdg goals/11.gif", use_container_width=True)
                    st.image("sdg goals/12.gif", use_container_width=True)
                with col5:
                    st.image("sdg goals/13.gif", use_container_width=True)
                    st.image("sdg goals/15.gif", use_container_width=True)
            elif label == "8 plastic\n":
                st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">The image is classified as PLASTIC, Which comes under "PLASTIC-RECYCLING"</p>
            </div>
            """, unsafe_allow_html=True)
                with col4:
                    st.image("sdg goals/3.gif", use_container_width=True)
                    st.image("sdg goals/6.gif", use_container_width=True)
                    st.image("sdg goals/11.gif", use_container_width=True)
                with col5:
                    st.image("sdg goals/12.gif", use_container_width=True)
                    st.image("sdg goals/13.gif", use_container_width=True)
                    st.image("sdg goals/14.gif", use_container_width=True)
            elif label == "9 Shoes\n":
                st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">The image is classified as SHOES, Which comes under "TEXTILE OR FOOTWEAR RECYCLING"</p>
            </div>
            """, unsafe_allow_html=True)
                with col4:
                    st.image("sdg goals/11.gif", use_container_width=True)
                    st.image("sdg goals/12.gif", use_container_width=True)
                with col5:
                    st.image("sdg goals/13.gif", use_container_width=True)
                    st.image("sdg goals/15.gif", use_container_width=True)
            elif label == "10 Trash\n":
                st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">The image is classified as TRASH, Which comes under "NON-RECYCLABLE OR GENERAL WASTE"</p>
            </div>
            """, unsafe_allow_html=True)
                with col4:
                    st.image("sdg goals/13.gif", use_container_width=True)
                    st.image("sdg goals/15.gif", use_container_width=True)
                with col5:
                    st.image("sdg goals/11.gif", use_container_width=True)
                    st.image("sdg goals/12.gif", use_container_width=True)
            elif label == "11 White-glassl\n":
                st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">The image is classified as WHITE-GLASS, Which comes under "GLASS-RECYCLING"</p>
            </div>
            """, unsafe_allow_html=True)
                with col4:
                    st.image("sdg goals/6.gif", use_container_width=True)
                    st.image("sdg goals/9.gif", use_container_width=True)
                    st.image("sdg goals/11.gif", use_container_width=True)
                with col5:
                    st.image("sdg goals/12.gif", use_container_width=True)
                    st.image("sdg goals/13.gif", use_container_width=True)
                    st.image("sdg goals/15.gif", use_container_width=True)                        
            else:
                st.error("The image is not classified as any relevant class.")

        with col3:
            st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.5); padding: 5px; border-radius: 70px;">
                <p style="text-align: center; color: black; font-weight: bold;">Information Related To Carbon Emission</p>
            </div>
            """, unsafe_allow_html=True)
            result = generate_carbon_footprint_info(label)
            st.success(result)