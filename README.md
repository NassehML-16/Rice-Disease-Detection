# Rice Disease Detector

Lightweight Streamlit app for rice leaf disease detection using a Keras/TensorFlow CNN model.

This repository contains:

- `app.py` — Streamlit application that loads a Keras `.h5` model, accepts an image upload or camera input, runs inference, shows predictions, and generates a downloadable PDF report.
- `rice_disease_cnn_model.h5` / `rice_disease_cnn_model_v2.h5` — example model files.
- `dataset/` — training / test folders (not used by the app at inference time).
- `requirements.txt` — Python dependencies used for the project.

Important behaviour
- Uploaded images are processed in memory and are not saved by the running app. The app generates a PDF report you can download.
- The app includes compatibility shims (in `app.py`) that help load older Keras `.h5` models that used keys such as `batch_shape` or legacy `DTypePolicy`. This makes the loader more tolerant to serialization differences across TF/Keras versions.

Requirements
- Python 3.11+ (matching your TensorFlow build)
- See `requirements.txt` for exact package versions. Typical requirements include:
  - streamlit
  - tensorflow (or tensorflow-cpu)
  - pillow
  - numpy

Quick setup (local)
1. Clone the repo.
2. Create and activate a virtual environment:
   - Windows (PowerShell):
     ```powershell
     python -m venv env
     .\env\Scripts\Activate.ps1
     ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run the app:
   ```powershell
   streamlit run app.py
   ```
5. Open the URL shown in the terminal 

Usage
- Navigate to the `Detection` page.
- Upload a rice leaf image (jpg/png) or use your webcam/mobile camera input.
- After prediction, view the top probabilities and click `Download Report` to get a detailed PDF with the image, predicted label, confidence, symptoms, and recommended controls.

Model notes and compatibility
- The app attempts to import the model loader from `tensorflow.keras.models` or (fallback) `keras.models`. If neither import works at static-analysis time, `app.py` falls back to calling `tf.keras.models.load_model` at runtime.
- If your `.h5` model fails to load due to serialization differences (errors mentioning `batch_shape` or `DTypePolicy`), `app.py` includes compatibility helpers:
  - `PatchedInputLayer` — accepts legacy `batch_shape` in InputLayer configs and maps it to `batch_input_shape`.
  - `LegacyDTypePolicy` — maps older dtype policy configs to the runtime `Policy` or returns a safe fallback.

If your model still fails to load:
- Check TensorFlow/Keras versions used to save the model. Re-saving the model with a matching TF/Keras version (or saving a SavedModel) often resolves compatibility issues.
- Convert the model locally with the same TF version used to train it.

Retraining guidance
- If you plan to collect labelled examples and retrain:
  - Maintain a CSV/JSON metadata file that records filename, label, confidence, timestamp and any notes.
  - Keep a well-structured directory layout: `collected_data/<Label_Name>/*.jpg`.
  - Preprocess new examples the same way as in `app.py` (resize to 224x224, scale to [0,1]).
  - Use data augmentation and class-balanced sampling when retraining to avoid bias.

Useful commands
- Open collected data folder (local):
  ```powershell
  start "" "${PWD}\\collected_data"
  ```
- Run unit test / quick inference from Python shell:
  ```python
  from PIL import Image
  import numpy as np
  from tensorflow.keras.models import load_model

  model = load_model('rice_disease_cnn_model.h5', compile=False)
  img = Image.open('example.jpg').convert('RGB').resize((224,224))
  x = np.expand_dims(np.array(img)/255.0, 0).astype('float32')
  preds = model.predict(x)
  print(preds)
  ```

Contributing
- Fixes and enhancements are welcome. Please open issues for bugs or feature requests and submit pull requests with clear descriptions.

License
- Add your preferred license. (Consider MIT for permissive use.)

Contact
- For questions about the code or model compatibility, open an issue in this repository.
