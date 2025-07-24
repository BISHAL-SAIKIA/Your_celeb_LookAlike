# 🔥 Your Celeb Look Alike - Deep Learning & Streamlit Project

## 📊 Objective

Build a web app that:

* Accepts a user-uploaded photo
* Detects the user's face
* Extracts facial features
* Compares features with a celebrity image dataset
* Displays the most similar-looking celebrity

---

## 📆 Tech Stack

* **Deep Learning**: VGGFace with ResNet50 backbone
* **Face Detection**: MTCNN
* **Feature Extraction**: Keras VGGFace
* **Similarity Metric**: Cosine Similarity (from scikit-learn)
* **Web Interface**: Streamlit
* **Preprocessing**: OpenCV, PIL, keras-vggface
* **Data Serialization**: Pickle

---

## 📂 Data Preparation (main.py)

### ✔️ 1. List Files

```python
actors = os.listdir('data')
```

Scans the `data/` folder containing subfolders of celebrity images.

### ✔️ 2. Create Filenames List

```python
filenames.append(os.path.join('data', actor, file))
```

Saves all image paths in `filenames.pkl`.

### ✔️ 3. Extract Image Features

```python
result = model.predict(preprocessed_input).flatten()
```

Uses **VGGFace (ResNet50)** to extract 2048-d feature embeddings.

### ✔️ 4. Save Embeddings

All embeddings are saved into `embeddings.pkl` for future similarity search.

---

## 🛠️ App Logic (app.py)

### ✔️ 1. Face Detection

```python
detector = MTCNN()
```

MTCNN locates the face coordinates from the uploaded image.

### ✔️ 2. Upload & Save Image

```python
st.file_uploader(...) and save_uploaded_image(...)
```

Uploads an image and saves it in `Uploads/` directory.

### ✔️ 3. Preprocess Image

```python
face = img[y:y + height, x:x + width]
```

Face is cropped using coordinates, resized to 224x224, and normalized.

### ✔️ 4. Feature Extraction

```python
result = model.predict(preprocessed_img).flatten()
```

VGGFace returns the feature vector (embedding) for the user's face.

### ✔️ 5. Recommend Most Similar Celebrity

```python
cosine_similarity(features.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0]
```

Cosine similarity is used to compare the user's face with all embeddings.

### ✔️ 6. Display Results

```python
st.columns(2)
```

* **Left Column**: Uploaded image
* **Right Column**: Most similar celebrity & their image

---

## 📊 Deep Learning Concepts

* **Transfer Learning**: Pretrained VGGFace (ResNet50) for feature extraction
* **Face Embeddings**: Representing facial features in high-dimensional vector space
* **Cosine Similarity**: Measuring similarity between user and celeb embeddings
* **Image Normalization**: Aligning input scale to match training conditions of the model

---

## 📁 Pickle Files

* `filenames.pkl`: Stores all celebrity image paths
* `embeddings.pkl`: Stores extracted 2048-d features for each celebrity image

---

## 🚜 How to Run

```bash
streamlit run app.py
```

Make sure you have `data/`, `Uploads/`, `filenames.pkl`, and `embeddings.pkl` in the project directory.

---
## 🔹 License

This project is for educational purposes. Use responsibly!
