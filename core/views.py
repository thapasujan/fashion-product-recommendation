import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import os

from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from .forms import ImageUploadForm

# Load precomputed image features and filenames
Image_features = pkl.load(open(os.path.join(settings.BASE_DIR, 'model/Images_features.pkl'), 'rb'))
filenames = pkl.load(open(os.path.join(settings.BASE_DIR, 'model/filenames.pkl'), 'rb'))

# Initialize ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# NearestNeighbors model for finding similar images
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

def recommend_images(request):
    recommended_images = []
    uploaded_image_url = None
    form = ImageUploadForm(request.POST or None, request.FILES or None)
    
    if request.method == 'POST' and form.is_valid():
        uploaded_image = form.cleaned_data['image']
        
        # Save the uploaded image to `media/uploads/`
        upload_path = os.path.join(settings.UPLOADS_DIR, uploaded_image.name)
        with default_storage.open(upload_path, 'wb') as f:
            f.write(uploaded_image.file.read())

        uploaded_image_url = os.path.join(settings.MEDIA_URL, 'uploads', uploaded_image.name)
        
        # Extract features and find nearest neighbors
        input_img_features = extract_features_from_images(upload_path, model)
        distances, indices = neighbors.kneighbors([input_img_features])
        
        # Collect recommended image URLs
        for idx in indices[0][1:]:  # Skip the first one (itself)
            recommended_images.append(os.path.join(settings.MEDIA_URL, filenames[idx]))

    return render(request, 'index.html', {
        'form': form,
        'uploaded_image_url': uploaded_image_url,
        'recommended_images': recommended_images,
    })
