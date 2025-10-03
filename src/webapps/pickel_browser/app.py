from flask import Flask, render_template, send_file, request, abort
import os
import io

from data_types.AgenticImage import AgenticImage
from utils.DataStorage import AgenticImageDataStorage

app = Flask(__name__)

TEMP_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../temp_output_flickr2k'))

# Hilfsfunktion: AgenticImage aus cpkl laden
def load_agenticimage_from_cpkl(path) -> AgenticImage:
    loaded_pickel = AgenticImageDataStorage.load_agentic_image(path)
    return loaded_pickel

@app.route('/')
def index():
    images = []
    for fname in os.listdir(TEMP_OUTPUT_DIR):
        if fname.endswith('.cpkl'):
            fpath = os.path.join(TEMP_OUTPUT_DIR, fname)
            try:
                agentic_image = load_agenticimage_from_cpkl(fpath)
                images.append({
                    'filename': fname,
                    'applied_transformers': agentic_image.applied_transformers,
                    'source_score': agentic_image.source_image.score if agentic_image.source_image else None,
                    'transformed_score': agentic_image.transformed_image.score if agentic_image.transformed_image else None
                })
            except Exception as e:
                images.append({'filename': fname, 'error': str(e)})
    return render_template('index.html', images=images)

@app.route('/image/<filename>')
def serve_image(filename):
    img_type = request.args.get('type', 'original')
    fpath = os.path.join(TEMP_OUTPUT_DIR, filename)
    if not os.path.exists(fpath):
        abort(404)
    try:
        agentic_image = load_agenticimage_from_cpkl(fpath)
        if img_type == 'original':
            img = agentic_image.source_image.get_image() if agentic_image.source_image else None
        elif img_type == 'transformed':
            img = agentic_image.transformed_image.get_image() if agentic_image.transformed_image else None
        else:
            abort(400)
        if img is None:
            abort(404)
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        abort(500, str(e))

if __name__ == '__main__':
    app.run(debug=True)
