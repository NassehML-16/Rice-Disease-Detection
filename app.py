# Import the necessary libraries 
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
from datetime import datetime
import csv

# Import Keras load_model with robust fallback to support environments
# where 'tensorflow.keras' isn't available to static analysis; prefer
# tensorflow.keras when possible, otherwise fallback to standalone keras.
try:
    # Import dynamically to avoid static analyzers complaining about the tensorflow.keras namespace.
    import importlib
    tf_keras_models = importlib.import_module('tensorflow.keras.models')
    keras_load_model = tf_keras_models.load_model
except Exception:
    try:
        import importlib
        keras_models = importlib.import_module('keras.models')
        keras_load_model = keras_models.load_model
    except Exception:
        # As a last resort, define a wrapper that uses tf.keras when
        # tensorflow is available at runtime; this keeps linters happy.
        def keras_load_model(path, compile=False, custom_objects=None):
            return tf.keras.models.load_model(path, compile=compile, custom_objects=custom_objects)

# ---------------------------
# Configuration / Constants
# ---------------------------
MODEL_PATH = "./rice_disease_cnn_model.h5"
IMG_SIZE = (224, 224)
CLASS_LABELS = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice', 'Leaf Smut', 'Tungro']

# Detailed disease info (cause, symptoms, prevention, control)
DISEASE_INFO = {
    'Brown Spot': {
        'cause': 'Fungus — Bipolaris oryzae (previously Helminthosporium/Cochliobolus).',
        'symptoms': 'Small circular brown lesions on leaves; may coalesce to large dead patches; reduced tillering and grain weight in severe cases.',
        'prevention_control': (
            '- Use certified seed and treat seeds before planting.\n'
            '- Avoid excess nitrogen fertiliser; maintain balanced nutrition.\n'
            '- Ensure good drainage and reduce leaf wetness periods.\n'
            '- Rotate crops and remove infected residue; use timely fungicide sprays if thresholds are exceeded.'
        )
    },
    'Bacterial Leaf Blight': {
        'cause': 'Bacterium — Xanthomonas oryzae pv. oryzae.',
        'symptoms': 'Water-soaked streaks on leaf margins that expand, yellowing and wilting; can cause severe yield loss.',
        'prevention_control': (
            '- Use resistant varieties and disease-free seed.\n'
            '- Avoid excess nitrogen and manage irrigation to reduce pathogen spread.\n'
            '- Practice good nursery hygiene and remove infected plants early.\n'
            '- Biological control and seed treatments may reduce inoculum; integrate management measures.'
        )
    },
    'Leaf Smut': {
        'cause': 'Fungus — Entyloma oryzae (leaf smut).',
        'symptoms': 'Small black or dark smutty spots on leaves; generally minor but can affect older leaves.',
        'prevention_control': (
            '- Avoid excessive nitrogen application.\n'
            '- Manage crop residue and weed hosts; maintain crop health.\n'
            '- Fungicide control is rarely necessary; seek local extension guidance if severe.'
        )
    },
    'Tungro': {
        'cause': 'Viral complex — RTBV and RTSV, transmitted by green leafhoppers.',
        'symptoms': 'Stunted plants, yellow-orange discoloration of leaves, reduced tillering and poor grain filling.',
        'prevention_control': (
            '- Use virus-free seed and resistant varieties where available.\n'
            '- Time planting to avoid peak leafhopper populations; remove volunteer plants and weeds.\n'
            '- Control vectors (leafhoppers) and destroy heavily infected fields early; focus on prevention.'
        )
    },
    'Healthy Rice': {
        'cause': 'No disease detected.',
        'symptoms': 'Green, healthy-looking leaf tissue without lesions or discoloration.',
        'prevention_control': ('- Maintain good agronomic practices: balanced fertilisation, clean seed, and proper water management.')
    }
}

# ---------------------------
# Helpers
# ---------------------------
# Compatibility shim to deserialize legacy keras 'DTypePolicy' objects saved in older model configs
class LegacyDTypePolicy:
    @classmethod
    def from_config(cls, config):
        # config is expected to contain a 'name' like 'float32' or be nested
        name = None
        if isinstance(config, dict):
            name = config.get('name')
            # sometimes config may be nested under 'config'
            if name is None and 'config' in config and isinstance(config['config'], dict):
                name = config['config'].get('name')
        if not name:
            name = 'float32'
        # Map to the available Policy class on this TensorFlow version
        Policy = getattr(tf.keras.mixed_precision, 'Policy', None)
        if Policy is None:
            exp = getattr(tf.keras.mixed_precision, 'experimental', None)
            Policy = getattr(exp, 'Policy', None) if exp is not None else None
        try:
            if Policy is not None:
                return Policy(name)
        except Exception:
            pass
        # Fallback: return string dtype
        return name


@st.cache_resource
def load_model_cached(path=MODEL_PATH):
    """
    Load Keras model with compatibility fixes for older saved models that used
    'batch_shape' in InputLayer config and legacy DTypePolicy objects.
    """
    try:
        # Try standard load first
        model = keras_load_model(path, compile=False)
        return model
    except Exception as e:
        # Attempt to load with compatibility custom_objects
        custom = {
            'InputLayer': PatchedInputLayer,
            'DTypePolicy': LegacyDTypePolicy,
        }
        # Try with custom_objects fallback
        try:
            model = keras_load_model(path, compile=False, custom_objects=custom)
            return model
        except Exception:
            # Re-raise the original exception to preserve traceback for debugging
            raise e


# Patched InputLayer to accept legacy 'batch_shape' key during deserialization
class PatchedInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, **kwargs):
        # Convert legacy 'batch_shape' to 'batch_input_shape' if present
        if 'batch_shape' in kwargs:
            # kwargs['batch_shape'] may be a list like [None, h, w, c]
            kwargs['batch_input_shape'] = tuple(kwargs.pop('batch_shape'))
        super().__init__(*args, **kwargs)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert PIL image to model-ready tensor: RGB, resized to IMG_SIZE, scaled to [0,1]."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


def predict_image(model, image: Image.Image):
    x = preprocess_image(image)
    preds = model.predict(x)
    probs = preds[0]
    idx = int(np.argmax(probs))
    label = CLASS_LABELS[idx]
    confidence = float(probs[idx])
    return label, confidence, probs


def create_report_pdf(image: Image.Image, label: str, confidence: float, notes: str = "") -> bytes:
    """Create a simple one-page PDF report (A4-like) using PIL and return bytes.

    Title is rendered bold and headings/subheadings are bold. Font sizes are increased
    for readability and the uploaded image is embedded in the report.
    """
    # Create a white canvas (A4 approximation: 2480x3508 px at 300dpi; we use smaller for speed)
    W, H = 1240, 1754  # roughly A4 at 150 dpi
    canvas = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)

    # Larger fonts for improved readability
    TITLE_SIZE = 48
    HEADING_SIZE = 22
    BODY_SIZE = 18

    # Load fonts; fall back to default and emulate bold if necessary
    try:
        font_title = ImageFont.truetype('DejaVuSans-Bold.ttf', TITLE_SIZE)
        font_heading = ImageFont.truetype('DejaVuSans-Bold.ttf', HEADING_SIZE)
        font_body = ImageFont.truetype('DejaVuSans.ttf', BODY_SIZE)
        use_emulated_bold = False
    except Exception:
        font_title = ImageFont.load_default()
        font_heading = ImageFont.load_default()
        font_body = ImageFont.load_default()
        use_emulated_bold = True

    margin = 40
    y = margin

    # Draw bold title (emulate bold if truetype not available)
    title_text = 'Rice Disease Detection Report'
    if use_emulated_bold:
        draw.text((margin, y), title_text, font=font_title, fill='black')
        draw.text((margin+1, y+1), title_text, font=font_title, fill='black')
    else:
        draw.text((margin, y), title_text, font=font_title, fill='black')
    y += TITLE_SIZE + 12

    # Insert metadata using larger body font
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    draw.text((margin, y), f'Date: {now}', font=font_body, fill='black')
    y += BODY_SIZE + 6
    draw.text((margin, y), f'Detection: {label}', font=font_body, fill='black')
    y += BODY_SIZE + 6
    draw.text((margin, y), f'Confidence: {confidence*100:.2f} %', font=font_body, fill='black')
    y += BODY_SIZE + 12

    # Disease description & control (multiline)
    info = DISEASE_INFO.get(label, {})
    desc = info.get('symptoms', '')
    ctl = info.get('prevention_control', '')

    # Symptoms heading (bold)
    if use_emulated_bold:
        draw.text((margin, y), 'Symptoms:', font=font_heading, fill='black')
        draw.text((margin+1, y+1), 'Symptoms:', font=font_heading, fill='black')
    else:
        draw.text((margin, y), 'Symptoms:', font=font_heading, fill='black')
    y += HEADING_SIZE + 6
    for line in desc.split('\n'):
        draw.text((margin+10, y), line, font=font_body, fill='black')
        y += BODY_SIZE - 2
    y += 8

    # Recommended Controls heading (bold)
    if use_emulated_bold:
        draw.text((margin, y), 'Recommended Controls:', font=font_heading, fill='black')
        draw.text((margin+1, y+1), 'Recommended Controls:', font=font_heading, fill='black')
    else:
        draw.text((margin, y), 'Recommended Controls:', font=font_heading, fill='black')
    y += HEADING_SIZE + 6
    for line in ctl.split('\n'):
        draw.text((margin+10, y), line, font=font_body, fill='black')
        y += BODY_SIZE - 2

    # Draw the (resized) image on the right side (larger to be clearer)
    img_w = 420
    img_h = 420
    thumb = image.copy()
    thumb.thumbnail((img_w, img_h))
    canvas.paste(thumb, (W - img_w - margin, margin + TITLE_SIZE))

    # Notes or user comments (bold heading)
    if notes:
        y_notes = max(y + 10, margin + TITLE_SIZE + img_h + 20 if img_h else y + 10)
        if use_emulated_bold:
            draw.text((margin, y_notes), 'Notes:', font=font_heading, fill='black')
            draw.text((margin+1, y_notes+1), 'Notes:', font=font_heading, fill='black')
        else:
            draw.text((margin, y_notes), 'Notes:', font=font_heading, fill='black')
        y_notes += HEADING_SIZE + 6
        for line in notes.split('\n'):
            draw.text((margin+10, y_notes), line, font=font_body, fill='black')
            y_notes += BODY_SIZE - 2

    # Save to bytes buffer as PDF
    buf = io.BytesIO()
    canvas.save(buf, format='PDF')
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

# ---------------------------
# App Layout
# ---------------------------
st.set_page_config(page_title='Rice Disease Detector', layout='wide')

st.sidebar.title('Rice Disease Detector')
page = st.sidebar.radio('Navigate', ['Detection', 'Information'])

# Problem statement
problem_statement = (
    "African smallholder farmers frequently lack rapid, local access to expert plant-disease diagnostics. "
    "This leads to delayed treatment, misuse of agrochemicals, and yield loss. A lightweight, accurate, "
    "and easy-to-use rice disease detection app can help farmers identify diseases early, recommend management, "
    "and reduce unnecessary pesticide use."
)

if page == 'Detection':
    st.title('Rice Disease Detector')
    st.markdown('### Problem Statement')
    st.write(problem_statement)

    st.markdown('---')
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader('Input Image')
        uploaded_file = st.file_uploader('Upload a rice leaf image (jpg, png)', type=['jpg', 'jpeg', 'png'])
        camera_file = st.camera_input('Or take a photo (mobile / webcam)')

        # Prefer camera input if provided
        input_image = None
        if camera_file is not None:
            input_bytes = camera_file.getvalue()
            input_image = Image.open(io.BytesIO(input_bytes))
        elif uploaded_file is not None:
            input_bytes = uploaded_file.read()
            input_image = Image.open(io.BytesIO(input_bytes))

        user_notes = st.text_area('Notes (optional)', help='Add context like field location, cultivar, or observations.')

        if input_image is not None:
            st.image(input_image, caption='Uploaded image (will not be saved)', use_container_width=True)

    with col2:
        st.subheader('Prediction')
        if input_image is None:
            st.info('Upload an image or use the camera to get a prediction.')
        else:
            # Load model
            try:
                model = load_model_cached()
            except Exception as e:
                st.error(f'Failed to load model: {e}')
                st.stop()

            with st.spinner('Running model...'):
                label, confidence, probs = predict_image(model, input_image)

            st.markdown(f'**Predicted:** {label}')
            st.markdown(f'**Confidence:** {confidence*100:.2f} %')

            # Show top-3 probabilities
            top_idx = np.argsort(probs)[::-1][:3]
            st.markdown('**Top predictions:**')
            for i in top_idx:
                st.write(f'{CLASS_LABELS[i]} — {probs[i]*100:.2f}%')

            # Disease details
            info = DISEASE_INFO.get(label, {})
            st.markdown('**Symptoms:**')
            st.write(info.get('symptoms', ''))
            st.markdown('**Recommended control measures:**')
            st.write(info.get('prevention_control', ''))

            # Create detailed PDF report and offer for download (no saving of user data)
            pdf_bytes = create_report_pdf(input_image, label, confidence, notes=user_notes)
            st.download_button(
                label='Download Report',
                data=pdf_bytes,
                file_name=f'rice_disease_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
                mime='application/pdf'
            )

            st.markdown(
    '<span style="color:gray; opacity:0.65">Note: Uploaded images are processed in memory and not saved by this app.</span>',
    unsafe_allow_html=True
)
            

elif page == 'Information':
    st.title('Disease Information')
    st.markdown('This section contains detailed information on the diseases the model detects.')

    for key in CLASS_LABELS:
        st.subheader(key)
        info = DISEASE_INFO.get(key, {})
        st.markdown(f"**Cause:** {info.get('cause','')}")
        st.markdown(f"**Symptoms:** {info.get('symptoms','')}")
        st.markdown('**Prevention & Control:**')
        st.write(info.get('prevention_control',''))
        st.markdown('---')

    st.markdown('---')
    st.subheader('Guidance for Farmers')
    st.markdown(
        '- Use certified, clean seed and choose resistant varieties where available.\n'
        '- Avoid excessive nitrogen fertiliser; balance nutrition.\n'
        '- Monitor fields regularly, remove infected plants early, and manage weeds and volunteer hosts.\n'
        '- Consult local extension services for approved agrochemicals and region-specific recommendations.'
    )

