import os
import glob
from deepface import DeepFace
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from PIL import Image, ImageDraw
from flask import Flask, render_template, url_for, send_from_directory, request, redirect, flash, Response, jsonify, session
import zipfile
import shutil
import time
import logging

# --- App Constants ---
# We no longer need this as the model is specified in the DeepFace call
# FACE_DETECTION_MODEL = "hog" 

# --- Flask App Setup ---
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
# Add a secret key for flashing messages
app.secret_key = 'a-super-secret-key-that-you-should-change'

# Use the /tmp directory for all generated content, as it's guaranteed to be writable on most platforms.
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['PROCESSED_FOLDER'] = '/tmp/processed'
app.config['THUMBNAIL_FOLDER'] = '/tmp/thumbnails'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# Global variable to store processed results to avoid reprocessing on every request (for POC)
processed_data_for_web = {}

# Ensure all temporary directories exist when the app starts.
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['THUMBNAIL_FOLDER'], exist_ok=True)

@app.before_request
def log_request_info():
    """Log information about the incoming request before handling it."""
    app.logger.debug(f"--- [DEBUG] Incoming Request: {request.method} {request.path} ---")

def clear_all_data():
    """Clears all generated files and resets the global data cache."""
    global processed_data_for_web
    processed_data_for_web = {}
    
    folders_to_clear = [
        app.config['UPLOAD_FOLDER'],
        app.config['PROCESSED_FOLDER'],
        app.config['THUMBNAIL_FOLDER']
    ]
    for folder in folders_to_clear:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    logging.info("Cleared all previous data and directories.")

def get_image_files_from_dir(directory):
    """
    Scans a directory RECURSIVELY for common image files.
    This handles zip files that have images inside a sub-folder.
    """
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png', 'gif', 'heic', 'heif']:
        pattern = os.path.join(directory, f'**/*.{ext.lower()}')
        image_paths.extend(glob.glob(pattern, recursive=True))
        pattern_upper = os.path.join(directory, f'**/*.{ext.upper()}')
        image_paths.extend(glob.glob(pattern_upper, recursive=True))

    logging.info(f"Found {len(image_paths)} images in {directory}")
    return sorted(list(set(image_paths)))

def cluster_faces(face_data, eps=0.45, min_samples=2, metric='euclidean'):
    """
    Clusters face encodings using DBSCAN to identify unique individuals.
    """
    if not face_data:
        app.logger.warning("No face data provided for clustering.")
        return None

    encodings = np.array([d['encoding'] for d in face_data])

    if encodings.size == 0:
        app.logger.warning("No encodings found in face_data.")
        return None

    app.logger.info(f"Clustering {len(encodings)} face encodings using DBSCAN with eps={eps}, min_samples={min_samples}, metric='{metric}'...")

    clt = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    clt.fit(encodings)
    labels = clt.labels_

    unique_labels = set(labels)
    num_unique_people = len(unique_labels) - (1 if -1 in labels else 0)
    num_noise_points = np.sum(np.array(labels) == -1)

    app.logger.info(f"Clustering complete. Found {num_unique_people} people and {num_noise_points} unclustered faces.")
    return labels

def draw_boxes_and_save(img_obj, locations, labels, original_path, output_dir, prefix=""):
    """Draws bounding boxes and labels on an image and saves it."""
    draw = ImageDraw.Draw(img_obj)
    for (top, right, bottom, left), label in zip(locations, labels):
        box_color = "yellow" if label == "Unclustered" else "lime"
        draw.rectangle(((left, top), (right, bottom)), outline=box_color, width=3)
        if label != "Unclustered":
            draw.text((left + 6, bottom - 15), str(label), fill='white')

    base_name = os.path.basename(original_path)
    prefix_str = f"{prefix}_" if prefix else ""
    output_filename = f"{prefix_str}{base_name}"
    output_path = os.path.join(output_dir, output_filename)
    
    img_obj.save(output_path)
    return output_filename

def crop_save_and_get_best_thumbnail(face_data_list, person_name, thumbnails_folder):
    """
    Crops the best face thumbnail for a person and returns its path.
    This version is memory-efficient, reading the image from disk.
    """
    if not face_data_list:
        return None

    # Find the face with the largest bounding box area
    best_face = max(face_data_list, key=lambda face: (face['location'][2] - face['location'][0]) * (face['location'][1] - face['location'][3]))
    
    # Memory-Efficient: Open and resize the image for the best face just in time
    with Image.open(best_face['image_path']).convert('RGB') as img:
        max_size = 1600
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size))

        # Define the box for cropping
        box = (best_face['location'][3], best_face['location'][0], best_face['location'][1], best_face['location'][2])
        cropped_img = img.crop(box)
        
        thumbnail_filename = f"thumb_{person_name}.jpg"
        thumbnail_path = os.path.join(thumbnails_folder, thumbnail_filename)
        cropped_img.save(thumbnail_path)
    
    return thumbnail_filename

def prepare_web_output(all_face_data, cluster_labels, output_dir_abs):
    """
    Organizes clustered data for rendering. Memory-efficient version that reads
    images from disk as needed instead of holding them in memory.
    """
    people = {}
    unclustered = []
    
    # Group all face data by their original image path and cluster label
    label_to_faces = defaultdict(list)
    if all_face_data:
        for i, data_point in enumerate(all_face_data):
            label = cluster_labels[i]
            label_to_faces[label].append(data_point)

    # --- Process Unclustered Faces ---
    if -1 in label_to_faces:
        # Group unclustered faces by their image path
        unclustered_by_image = defaultdict(list)
        for face in label_to_faces[-1]:
            unclustered_by_image[face['image_path']].append(face['location'])
        
        # Draw boxes for each image that has unclustered faces
        for path, locations in unclustered_by_image.items():
            with Image.open(path).convert('RGB') as img:
                max_size = 1600
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size))
                
                filename = draw_boxes_and_save(
                    img, locations, ["Unclustered"] * len(locations),
                    path, output_dir_abs, prefix="unclustered"
                )
                unclustered.append(filename)

    # --- Process Clustered Faces ---
    for label_id, faces_in_cluster in label_to_faces.items():
        if label_id == -1:
            continue

        person_id = f"Person_{label_id + 1}"
        people[person_id] = {'images': [], 'thumbnail': None}
        
        # Group faces for this person by their original image path
        clustered_by_image = defaultdict(list)
        for face in faces_in_cluster:
            clustered_by_image[face['image_path']].append(face['location'])

        # Draw boxes for each image associated with this person
        for path, locations in clustered_by_image.items():
            with Image.open(path).convert('RGB') as img:
                max_size = 1600
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size))

                filename = draw_boxes_and_save(
                    img, locations, [person_id] * len(locations),
                    path, output_dir_abs, prefix=person_id
                )
                people[person_id]['images'].append(filename)

        # Generate a thumbnail for this person using the most suitable face
        thumbnail_filename = crop_save_and_get_best_thumbnail(faces_in_cluster, person_id, app.config['THUMBNAIL_FOLDER'])
        if thumbnail_filename:
            people[person_id]['thumbnail'] = thumbnail_filename
            
    return people, sorted(list(set(unclustered)))

# --- Routes for serving generated files ---
# These routes are necessary because the files are now stored in /tmp, 
# outside the static folder that Flask serves by default.

@app.route('/processed/<filename>')
def serve_processed_image(filename):
    """Serves an image with bounding boxes drawn on it."""
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/thumbnails/<filename>')
def serve_thumbnail(filename):
    """Serves a cropped face thumbnail."""
    return send_from_directory(app.config['THUMBNAIL_FOLDER'], filename)


# --- Main Processing Pipeline (as a Generator) ---
def run_pipeline_and_yield_progress(eps_value):
    """
    Runs the entire face processing pipeline and yields Server-Sent Events (SSE)
    formatted progress updates.
    """
    global processed_data_for_web
    
    def sse_message(data):
        """Formats a dictionary into an SSE message string."""
        import json
        return f"data: {json.dumps(data)}\n\n"

    try:
        # Use the absolute paths directly from the app config.
        input_dir_abs_path = app.config['UPLOAD_FOLDER']
        output_dir_abs_path = app.config['PROCESSED_FOLDER']

        yield sse_message({"message": "Starting new analysis..."})

        # 1. Unzip and find images
        # Construct the path to the uploaded zip file directly.
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.zip')
        if not os.path.exists(zip_path):
            yield sse_message({"error": "No zip file found. Please upload again."})
            return

        yield sse_message({"message": "Extracting images from zip file..."})
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(input_dir_abs_path)
        
        image_paths = get_image_files_from_dir(input_dir_abs_path)
        if not image_paths:
            yield sse_message({"error": "No images found in the zip file."})
            return

        # 2. Face Detection and Encoding
        all_face_data = []
        total_images = len(image_paths)
        for i, img_path in enumerate(image_paths):
            yield sse_message({"message": f"Processing image {i+1}/{total_images}: {os.path.basename(img_path)}", "progress": int(((i+1)/total_images)*100)})
            try:
                # Open image and resize if it's large to speed up processing
                with Image.open(img_path).convert('RGB') as img:
                    max_size = 1600
                    if img.width > max_size or img.height > max_size:
                        img.thumbnail((max_size, max_size))

                    # Convert PIL image to numpy array for deepface
                    img_np = np.array(img)

                    # Use DeepFace to find faces
                    # We set detector_backend to 'retinaface' which is often a good balance.
                    # Other options: 'opencv', 'ssd', 'dlib', 'mtcnn'
                    # The 'ArcFace' model is used for recognition (creating the embedding)
                    found_faces = DeepFace.represent(
                        img_path=img_np,
                        model_name='ArcFace',
                        detector_backend='retinaface',
                        enforce_detection=False
                    )

                    for face_info in found_faces:
                        if face_info["face_confidence"] > 0:  # Check if a face was actually found
                            # Memory-Efficient: Do NOT store the image object here.
                            # Only store the path and metadata.
                            all_face_data.append({
                                'image_path': img_path,
                                'encoding': face_info['embedding'],
                                'location': (face_info['facial_area']['y'], face_info['facial_area']['x'] + face_info['facial_area']['w'], face_info['facial_area']['y'] + face_info['facial_area']['h'], face_info['facial_area']['x']),
                            })

            except Exception as e:
                app.logger.error(f"Error processing image {img_path}: {e}")
                yield sse_message({"error": f"Failed to process {os.path.basename(img_path)}."})
        
        if not all_face_data:
            yield sse_message({"error": "Could not detect any faces in the uploaded images."})
            return

        # 3. Face Clustering
        yield sse_message({"message": f"Found {len(all_face_data)} faces. Now clustering..."})
        cluster_labels = cluster_faces(all_face_data, eps=eps_value)

        # 4. Prepare output for web
        yield sse_message({"message": "Organizing results and generating thumbnails..."})
        people, unclustered = prepare_web_output(all_face_data, cluster_labels, output_dir_abs_path)
        
        processed_data_for_web = {
            'people': people,
            'unclustered': unclustered,
            'eps_used': eps_value
        }

        yield sse_message({"message": "Analysis complete!", "progress": 100, "status": "complete"})

    except Exception as e:
        app.logger.error(f"An error occurred in the pipeline: {e}", exc_info=True)
        yield sse_message({"error": "A critical error occurred during processing."})


# --- Main App Routes ---

@app.route('/', methods=['GET'])
def index():
    people = processed_data_for_web.get('people', {})
    unclustered = processed_data_for_web.get('unclustered', [])
    
    # Generate correct URLs using the new routes for serving files from /tmp.
    people_with_urls = {
        person_id: {
            'images': [url_for('serve_processed_image', filename=img) for img in data['images']],
            'thumbnail': url_for('serve_thumbnail', filename=data['thumbnail']) if data.get('thumbnail') else None
        }
        for person_id, data in people.items()
    }
    unclustered_urls = [url_for('serve_processed_image', filename=img) for img in unclustered]

    return render_template(
        'index.html',
        people=people_with_urls,
        unclustered_images=unclustered_urls,
        model_name='ArcFace',
        eps_value=processed_data_for_web.get('eps_used', 0.5)
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'zipfile' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['zipfile']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and file.filename.endswith('.zip'):
        clear_all_data()
        
        # Save the zip file to a predictable, fixed location.
        filename = "input.zip" 
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(zip_path)
        
        # We no longer need the session. The processing stream will find the file.
        
        # Redirect to the processing page
        return redirect(url_for('processing'))
    else:
        flash('Invalid file type. Please upload a .zip file.')
        return redirect(request.url)

@app.route('/processing')
def processing():
    # This page just shows the progress bar and initiates the SSE stream
    return render_template('processing.html')

@app.route('/stream')
def stream():
    eps = request.args.get('eps', 0.45, type=float)
    return Response(run_pipeline_and_yield_progress(eps_value=eps), mimetype='text/event-stream')

# --- Main execution for Flask app ---
if __name__ == '__main__':
    # Setting debug=True enables auto-reloading and gives helpful error pages
    # NOTE: use_reloader=False is important for the upload process to not get stuck
    app.run(debug=True, use_reloader=False)
