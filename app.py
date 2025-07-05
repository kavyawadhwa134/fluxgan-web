from flask import Flask, render_template, request, jsonify
from fluxgan_flux_only import load_generator_model, predict_flux
import torch

app = Flask(__name__)

# Load model on start
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = load_generator_model("checkpoint_65000.tar", device)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        enrichment = float(request.form['enrichment'])
        flux = predict_flux(enrichment, generator, device)
        if flux is None:
            return jsonify({'error': 'No matching flux values found.'})
        return jsonify({
            'enrichment': enrichment,
            'flux': f"{flux:.4e} n/cm²-s"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
