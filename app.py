from flask import Flask, render_template, request
import numpy as np
import pickle
from PIL import Image
import base64
import io

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        img = data.get('image')

        img = img.split(',')[1]

        image = Image.open(io.BytesIO(base64.b64decode(img))).convert('L')
        image = image.resize((28, 28))

        # Convert to numpy
        img_array = np.array(image)

# Invert colors (MNIST: white digit on black)
        img_array = 255 - img_array


# Flatten
        final_img = img_array.reshape(1, -1)
        # Invert (MNIST style)
        img_array = 255 - img_array

# Normalize
        img_array = img_array / 255.0

# Remove noise
        img_array[img_array < 0.2] = 0

# Find bounding box
        coords = np.argwhere(img_array > 0)
        if coords.size == 0:
         return {'prediction': "No digit found"}

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        digit = img_array[y0:y1, x0:x1]

# Resize to 20x20
        digit_img = Image.fromarray((digit * 255).astype(np.uint8))
        digit_img = digit_img.resize((20, 20))

# Center in 28x28
        new_img = np.zeros((28, 28))
        new_img[4:24, 4:24] = np.array(digit_img) / 255.0

        final_img = new_img.reshape(1, -1)

        prediction = model.predict(final_img)[0]
        confidence = max(model.predict_proba(final_img)[0])

        print("Prediction:", prediction, "Confidence:", confidence)

        if confidence < 0.3:
            return {'prediction': "No digit found"}

        return {'prediction': int(prediction)}

    except Exception as e:
        print("ERROR:", e)
        return {'prediction': "Error occurred"}

if __name__ == "__main__":
    app.run(debug=True)