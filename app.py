import numpy as np
import pickle as pkl
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer

st.header('Fashion Recommendation System')

# Load precomputed image features and filenames
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

def generate_caption(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(384, 384))  # BLIP model expects 384x384 images
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = processor(images=img_expand_dim, return_tensors="pt")

    # Generate caption with increased beam search size for more detailed captions
    outputs = caption_model.generate(**img_preprocess, num_beams=5, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def highlight_differences(captions):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(captions)
    terms = vectorizer.get_feature_names_out()
    differences = []
    
    for i in range(len(captions)):
        tfidf_scores = tfidf_matrix[i].toarray()[0]
        important_terms = [(terms[j], tfidf_scores[j]) for j in np.argsort(tfidf_scores) if tfidf_scores[j] > 0]
        important_terms.sort(key=lambda x: x[1], reverse=True)
        differences.append([term for term, score in important_terms[:3]])
    
    return differences

def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

# Initialize the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([
    model,
    GlobalMaxPool2D()
])

# Initialize the Nearest Neighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Initialize the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Ensure the 'upload' directory exists
upload_dir = 'upload'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Handle file upload
upload_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if upload_file is not None:
    # Save the uploaded file
    file_path = os.path.join(upload_dir, upload_file.name)
    with open(file_path, 'wb') as f:
        f.write(upload_file.getbuffer())
    
    st.subheader('Uploaded Image')
    st.image(upload_file)

    # Extract features from the uploaded image
    input_img_features = extract_features_from_images(file_path, model)

    # Find the nearest neighbors
    distances, indices = neighbors.kneighbors([input_img_features])

    st.subheader('Recommended Images and Descriptions')
    cols = st.columns(5)
    base_path = '/Users/arunima/Desktop/archive'
    
    captions = []
    image_paths = []

    for i, col in enumerate(cols):
        try:
            image_rel_path = filenames[indices[0][i+1]]
            image_path = os.path.join(base_path, image_rel_path)
            if not os.path.isabs(image_path):
                image_path = os.path.abspath(image_path)
            
            # Generate caption for the image
            caption = generate_caption(image_path)
            captions.append(caption)
            image_paths.append(image_path)
        except Exception as e:
            st.error(f"Error displaying image {filenames[indices[0][i+1]]}: {e}")

    differences = highlight_differences(captions)
    
    for i, col in enumerate(cols):
        with col:
            st.image(image_paths[i])
            st.write(f"Caption: {captions[i]}")
            st.write(f"Unique features: {', '.join(differences[i])}")
