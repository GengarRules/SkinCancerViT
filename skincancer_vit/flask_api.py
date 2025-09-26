import io
import base64
import traceback
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

from skincancer_vit.model import SkinCancerViTModel
from skincancer_vit.utils import get_torch_device
from skincancer_vit.xai_utils import get_attention_map_output_gradcam


HF_MODEL_REPO = "ethicalabs/SkinCancerViT"
DEVICE = get_torch_device()

app = Flask(__name__)
CORS(app)

# Lazy-loaded model holder
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        print(f"Loading model {HF_MODEL_REPO} to {DEVICE}...")
        _MODEL = SkinCancerViTModel.from_pretrained(HF_MODEL_REPO)
        _MODEL.to(DEVICE)
        _MODEL.eval()
        print("Model loaded.")
    return _MODEL


@app.route("/api/predict", methods=["POST"])
def predict():
    """Accepts multipart/form-data with fields:
    - image: file
    - age: integer
    - localization: string

    Returns JSON with prediction, confidence and a base64-encoded CAM PNG under 'cam'.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        image_file = request.files["image"]
        age = request.form.get("age")
        localization = request.form.get("localization", "unknown")

        # Basic validation and parsing
        try:
            age_val = int(age) if age is not None else None
        except Exception:
            return jsonify({"error": "Invalid age value."}), 400

        # Open image
        image = Image.open(image_file.stream).convert("RGB")

        model = get_model()

        # Run prediction
        predicted_dx, confidence = model.full_predict(
            raw_image=image, raw_age=age_val, raw_localization=localization, device=DEVICE
        )

        # Get class idx for CAM (model.config.label2id expects label->id mapping)
        target_class_idx = model.config.label2id.get(predicted_dx, None)

        cam_b64 = None
        if target_class_idx is not None:
            try:
                cam_np = get_attention_map_output_gradcam(
                    full_multimodal_model=model,
                    image_input=image,
                    target_class_idx=target_class_idx,
                    img_size=(224, 224),
                    cam_method=None,  # allow default inside util
                    patch_size=16,
                    raw_age=age_val,
                    raw_localization=localization,
                    device=DEVICE,
                )

                # cam_np may be a PIL Image, numpy array, or similar. Convert to PNG bytes.
                if isinstance(cam_np, Image.Image):
                    cam_img = cam_np
                else:
                    # try to import numpy and build image
                    try:
                        import numpy as _np

                        cam_img = Image.fromarray(_np.uint8(cam_np))
                    except Exception:
                        cam_img = None

                if cam_img is not None:
                    buf = io.BytesIO()
                    cam_img.save(buf, format="PNG")
                    cam_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception:
                traceback.print_exc()
                # proceed without CAM if it fails

        return jsonify({
            "prediction": predicted_dx,
            "confidence": float(confidence),
            "cam": cam_b64,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local development only. In production use a WSGI server.
    app.run(host="0.0.0.0", port=8000, debug=True)
