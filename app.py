from flask import Flask, render_template, request, jsonify
from fluxgan_core import load_checkpoint, predict_flux
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        enrichment = request.form['enrichment']
        reactor = request.form['reactor']

        checkpoint_path = f'./fluxgan_model/{reactor}/checkpoint_flux_only.tar'
        if not os.path.exists(checkpoint_path):
            return jsonify({'error': f'Checkpoint not found for {reactor.upper()}'}), 500

        generator = load_checkpoint(checkpoint_path)
        flux = predict_flux(generator, enrichment)

        return jsonify({
            'reactor': reactor.upper(),
            'enrichment': float(enrichment),
            'flux': float(flux)
        })


    except Exception as e:
        print(f"❌ Error in /predict route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5050)
