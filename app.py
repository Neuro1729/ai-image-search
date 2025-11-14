import streamlit as st
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel
import torch
import pickle
import numpy as np
import faiss
import os

st.set_page_config(page_title="AI Image Search", layout="wide")
st.title("AI Image Search Engine 🖼️")

# ------------------ Load embeddings ------------------
@st.cache_data
def load_embeddings(pkl_path="embeddings.pkl"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    image_paths = data["paths"]
    embeddings = np.array(data["embeddings"]).astype("float32")
    return image_paths, embeddings

image_paths, embeddings = load_embeddings()

# ------------------ Build FAISS index ------------------
@st.cache_resource
def build_faiss(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

index = build_faiss(embeddings)

# ------------------ Load CLIP model ------------------
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

model, processor, device = load_clip()

# ------------------ Function to get embedding ------------------
def get_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy().astype("float32")

# ------------------ Streamlit UI ------------------
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Fix orientation and resize uploaded image
    query_image = Image.open(uploaded_file).convert("RGB")
    query_image = ImageOps.exif_transpose(query_image)  # fix rotation
    query_image_resized = query_image.resize((300, 300))
    st.image(query_image_resized, caption="Uploaded Image", width=300)

    # Get embedding & search
    query_emb = get_embedding(query_image)
    k = 4  # number of similar images
    distances, indices = index.search(query_emb, k)

    st.subheader("Top 4 similar images:")

    # Display results in 2x2 grid
    top4_indices = indices[0]
    for row in range(2):
        cols = st.columns(2,gap="small")
        for col_idx in range(2):
            idx = top4_indices[row*2 + col_idx]
            img_path = image_paths[idx]
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = img.rotate(-90, expand=True)
                cols[col_idx].image(img, width=220)  # adjust size
            else:
                cols[col_idx].write("Image not found")
