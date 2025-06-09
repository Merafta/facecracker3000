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

# THE FIX (1/2): Consolidate and clarify folder configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['THUMBNAIL_FOLDER'] = 'static/thumbnails'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# Global variable to store processed results to avoid reprocessing on every request (for POC)
processed_data_for_web = {}
# Ensure all directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['THUMBNAIL_FOLDER'], exist_ok=True)

@app.before_request
def log_request_info():
    """Log information about the incoming request before handling it."""
    # This will print to the terminal for every request the server receives.
    print(f"--- [DEBUG] Incoming Request: {request.method} {request.path} ---")

def clear_all_data():
    """Clears all generated files and resets the global data cache."""
    global processed_data_for_web
    processed_data_for_web = {}
    
    # THE FIX: Use the correct folder configuration keys.
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
    THE FIX (2/2): Scans a directory RECURSIVELY for common image files.
    This handles zip files that have images inside a sub-folder.
    """
    image_paths = []
    # Use recursive glob to find files in subdirectories
    for ext in ['jpg', 'jpeg', 'png', 'gif']:
        pattern = os.path.join(directory, f'**/*.{ext.lower()}')
        image_paths.extend(glob.glob(pattern, recursive=True))
        # Also check for uppercase extensions
        pattern_upper = os.path.join(directory, f'**/*.{ext.upper()}')
        image_paths.extend(glob.glob(pattern_upper, recursive=True))

    logging.info(f"Found {len(image_paths)} images in {directory}")
    return sorted(list(set(image_paths)))

def cluster_faces(face_data, eps=0.45, min_samples=2, metric='euclidean'):
    """
    Clusters face encodings using DBSCAN to identify unique individuals.

    Args:
        face_data (list): A list of dictionaries, where each dictionary contains
                          at least an 'encoding' key with a face encoding.
        eps (float): The maximum distance between samples for one to be
                     considered as in the neighborhood of the other (DBSCAN param).
        min_samples (int): The number of samples in a neighborhood for a point
                           to be considered as a core point (DBSCAN param).
        metric (str): The distance metric to use for clustering.

    Returns:
        numpy.ndarray: An array of cluster labels for each face encoding.
                       Label -1 indicates a noise point (unclustered face).
                       Returns None if no encodings are provided.
    """
    if not face_data:
        print("No face data provided for clustering.")
        return None

    # Extract encodings from the face_data list
    encodings = np.array([d['encoding'] for d in face_data])

    if encodings.size == 0:
        print("No encodings found in face_data.")
        return None

    print(f"Clustering {len(encodings)} face encodings using DBSCAN with eps={eps}, min_samples={min_samples}, metric='{metric}'...")

    # Initialize DBSCAN
    clt = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    clt.fit(encodings)

    # Get the cluster labels. Each face encoding will have a label.
    # Faces with the same label are considered to be the same person.
    # Label -1 means it's a noise point (couldn't be clustered).
    labels = clt.labels_

    # Determine the number of unique clusters (people)
    # We subtract 1 if -1 (noise) is present in the labels
    unique_labels = set(labels)
    num_unique_people = len(unique_labels) - (1 if -1 in labels else 0)
    num_noise_points = np.sum(np.array(labels) == -1)

    print(f"Clustering complete.")
    print(f"  Found {num_unique_people} unique people (clusters)." )
    print(f"  Found {num_noise_points} noise points (unclustered faces).")

    return labels

def draw_boxes_and_save(img_obj, locations, labels, original_path, output_dir, prefix=""):
    """Draws bounding boxes and labels on an image and saves it with a prefix."""
    draw = ImageDraw.Draw(img_obj)
    for (top, right, bottom, left), label in zip(locations, labels):
        # Use yellow for unclustered, lime for clustered people
        box_color = "yellow" if label == "Unclustered" else "lime"
        draw.rectangle(((left, top), (right, bottom)), outline=box_color, width=3)
        # Only add a text label if it's not a generic unclustered one
        if label != "Unclustered":
            draw.text((left + 6, bottom - 15), str(label), fill='white')

    base_name = os.path.basename(original_path)
    # Ensure prefix has a separator if it exists
    prefix_str = f"{prefix}_" if prefix else ""
    output_filename = f"{prefix_str}{base_name}"
    output_path = os.path.join(output_dir, output_filename)
    
    img_obj.save(output_path)
    return output_filename

def crop_save_and_get_best_thumbnail(face_data_list, person_name, thumbnails_folder):
    """Crops the best face thumbnail for a person and returns its path."""
    if not face_data_list:
        return None

    best_face = max(face_data_list, key=lambda face: (face['location'][2] - face['location'][0]) * (face['location'][1] - face['location'][3]))
    
    img = best_face['resized_image_obj']
    box = (best_face['location'][3], best_face['location'][0], best_face['location'][1], best_face['location'][2])
    
    cropped_img = img.crop(box)
    
    thumbnail_filename = f"thumb_{person_name}.jpg" # Consistent naming
    thumbnail_path = os.path.join(thumbnails_folder, thumbnail_filename)
    cropped_img.save(thumbnail_path)
    
    return thumbnail_filename

def prepare_web_output(all_face_data, cluster_labels, output_dir_abs):
    """Organizes clustered data for rendering in the web template."""
    people = {}
    unclustered = []
    
    label_to_faces = defaultdict(list)
    if all_face_data:
        for i, data_point in enumerate(all_face_data):
            label = cluster_labels[i]
            label_to_faces[label].append(data_point)

    # First, handle UNCLUSTERED faces
    if -1 in label_to_faces:
        unclustered_map = defaultdict(lambda: {'locations': [], 'img_obj': None})
        for face in label_to_faces[-1]:
            path = face['image_path']
            unclustered_map[path]['locations'].append(face['location'])
            unclustered_map[path]['img_obj'] = face['resized_image_obj']
        
        for path, data in unclustered_map.items():
            # Pass "Unclustered" as the label for drawing yellow boxes
            filename = draw_boxes_and_save(
                data['img_obj'].copy(), 
                data['locations'], 
                ["Unclustered"] * len(data['locations']), # Pass a list of labels
                path, 
                output_dir_abs,
                prefix="unclustered"
            )
            unclustered.append(filename)

    # Now, handle CLUSTERED people
    for label_id, faces_in_cluster in label_to_faces.items():
        if label_id == -1:
            continue

        person_id = f"Person_{label_id + 1}"
        people[person_id] = {'images': [], 'thumbnail': None}
        
        image_to_faces_map = defaultdict(lambda: {'locations': [], 'img_obj': None})
        for face in faces_in_cluster:
            path = face['image_path']
            image_to_faces_map[path]['locations'].append(face['location'])
            image_to_faces_map[path]['img_obj'] = face['resized_image_obj']

        for path, data in image_to_faces_map.items():
            filename = draw_boxes_and_save(
                data['img_obj'].copy(), 
                data['locations'], 
                [person_id] * len(data['locations']), # Pass person_id as the label
                path, 
                output_dir_abs,
                prefix=person_id
            )
            # THE FIX: Append the filename, not a broken path
            people[person_id]['images'].append(filename)

        thumbnail_filename = crop_save_and_get_best_thumbnail(faces_in_cluster, person_id, app.config['THUMBNAIL_FOLDER'])
        if thumbnail_filename:
            people[person_id]['thumbnail'] = thumbnail_filename
            
    return people, sorted(list(set(unclustered)))

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
        base_dir = os.path.abspath(os.path.dirname(__file__))
        input_dir_abs_path = os.path.join(base_dir, app.config['UPLOAD_FOLDER'])
        output_dir_abs_path = os.path.join(base_dir, app.config['PROCESSED_FOLDER'])

        yield sse_message({"message": "Starting new analysis..."})

        # 1. Unzip and find images
        zip_path = next(glob.iglob(os.path.join(app.config['UPLOAD_FOLDER'], '*.zip')), None)
        if not zip_path:
            logging.error("PIPELINE ERROR: No .zip file found in upload folder.")
            yield sse_message({"message": "Error: Could not find the uploaded .zip file.", "error": True})
            return
        
        unzip_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'unzipped')
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        
        image_files = get_image_files_from_dir(unzip_dir)
        
        if not image_files:
            logging.error(f"PIPELINE ERROR: get_image_files_from_dir returned 0 files from {unzip_dir}.")
            yield sse_message({"message": "Error: No valid image files found in the zip.", "error": True})
            return
        
        total_images = len(image_files)
        yield sse_message({"message": f"Found {total_images} images. Starting face analysis..."})
        
        # 2. Face Detection and Encoding Loop
        all_face_data = []
        total_faces_found = 0
        images_with_no_faces = []
        MAX_IMAGE_DIMENSION = 1600

        for i, image_path in enumerate(image_files):
            progress_percent = int(((i + 1) / total_images) * 100)
            msg = f"({i+1}/{total_images}) Analyzing {os.path.basename(image_path)}..."
            yield sse_message({"message": msg, "progress": progress_percent})

            # Define temp_image_path here to ensure it's in scope for the finally block
            temp_image_path = None
            try:
                # Load and potentially resize the image
                img_to_process = Image.open(image_path)
                if img_to_process.mode not in ['RGB', 'L']:
                    img_to_process = img_to_process.convert('RGB')
                if max(img_to_process.size) > MAX_IMAGE_DIMENSION:
                    yield sse_message({"message": f"{msg} Image is large, resizing...", "progress": progress_percent})
                    img_to_process.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
                
                # **THE FIX:** Save the (potentially resized) image to a temporary file.
                # This ensures DeepFace analyzes the same image we use for drawing later.
                temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{os.path.basename(image_path)}")
                img_to_process.save(temp_image_path)

                # Now, analyze the temporary (resized) image file.
                embedding_objs = DeepFace.represent(
                    img_path=temp_image_path, 
                    model_name='ArcFace', 
                    enforce_detection=False,
                    detector_backend='retinaface'
                )

                # The rest of the logic for processing `embedding_objs` is correct
                if embedding_objs and len(embedding_objs) > 0:
                    newly_found = 0
                    for emb_obj in embedding_objs:
                        facial_area = emb_obj['facial_area']
                        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                        
                        # **New Sanity Check:** If the bounding box is most of the image, it's probably a false positive.
                        img_width, img_height = img_to_process.size
                        box_area = w * h
                        img_area = img_width * img_height

                        # If box covers more than 95% of the image, skip it.
                        if box_area > 0 and img_area > 0 and (box_area / img_area > 0.95):
                            logging.warning(f"Skipping a bounding box on {os.path.basename(image_path)} that covers most of the image (likely a false positive).")
                            continue
                        
                        newly_found += 1
                        # Convert (x, y, w, h) to (top, right, bottom, left)
                        face_location = (y, x + w, y + h, x)
                        
                        all_face_data.append({
                            'image_path': image_path,
                            'location': face_location,
                            'encoding': emb_obj['embedding'],
                            'resized_image_obj': img_to_process
                        })
                    
                    if newly_found > 0:
                        total_faces_found += newly_found
                        msg_faces = f"Found {newly_found} face(s). Total so far: {total_faces_found}."
                        yield sse_message({"message": f"{msg} {msg_faces}", "progress": progress_percent})
                    else:
                        # This case happens if all found faces were too large (false positives)
                        images_with_no_faces.append(os.path.basename(image_path))
                        yield sse_message({"message": f"{msg} No valid faces found.", "progress": progress_percent})
                else:
                    # THE FIX: Add image to the no_faces list
                    images_with_no_faces.append(os.path.basename(image_path))
                    yield sse_message({"message": f"{msg} No faces found.", "progress": progress_percent})

            except Exception as e:
                logging.error(f"An unexpected error occurred processing {os.path.basename(image_path)}: {e}", exc_info=True)
                yield sse_message({"message": f"Error on {os.path.basename(image_path)}. See console.", "error": True})
                # continue is handled by the finally block
            
            finally:
                # **Crucially, clean up the temp file** whether an error occurred or not.
                if temp_image_path and os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        
        yield sse_message({"message": f"Finished image analysis. Found {total_faces_found} total faces. Now clustering..."})

        if not all_face_data:
            yield sse_message({"message": "No faces were detected in any images.", "error": True})
            return

        # THE FIX: No longer need to get from session, it's passed as an argument.
        yield sse_message({"message": f"Clustering with eps = {eps_value}..."})
        
        cluster_labels_arr = cluster_faces(all_face_data, eps=eps_value, min_samples=2, metric='cosine')

        yield sse_message({"message": "Clustering complete. Generating final output images..."})
        
        people_map, unclustered_list = prepare_web_output(all_face_data, cluster_labels_arr, output_dir_abs_path)
        
        # Update the global data structure with results AND the eps value used
        processed_data_for_web = {
            "people": people_map,
            "unclustered_faces": len(unclustered_list),
            "unclustered_images": unclustered_list,
            "total_people": len(people_map),
            "images_with_no_faces": images_with_no_faces,
            "eps_used": eps_value  # Store the eps value that was used for this result
        }
        yield sse_message({"message": "Done!", "progress": 100, "finished": True})

    except Exception as e:
        logging.error(f"A critical error occurred in the pipeline: {e}", exc_info=True)
        yield sse_message({"message": f"A critical error occurred: {e}", "error": True})

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # **THE FIX:** Ensure processed_data_for_web is a dictionary before accessing it.
    # This prevents the app from crashing on the first load if the data is None.
    current_data = processed_data_for_web if isinstance(processed_data_for_web, dict) else {}
    
    return render_template(
        'index.html', 
        data=current_data, 
        model_name='ArcFace',
        eps_value=current_data.get('eps_used', 0.5)
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'zipfile' not in request.files:
        return redirect(request.url)
    file = request.files['zipfile']
    if file.filename == '' or not file.filename.endswith('.zip'):
        return redirect(request.url)

    # THE FIX: Clear all old data BEFORE saving the new file.
    clear_all_data()

    try:
        eps_value = float(request.form.get('eps_value', '0.5'))
    except (ValueError, TypeError):
        eps_value = 0.5
    session['eps_value'] = eps_value

    zip_filename = file.filename
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_filename)
    file.save(zip_path)
    
    session['is_processing'] = True
    return redirect(url_for('processing'))

@app.route('/processing')
def processing():
    """
    This route shows a page that will then connect to the /stream
    endpoint to get live updates.
    """
    if not session.get('is_processing'):
        # If the user somehow lands here without uploading, send them home.
        return redirect(url_for('index'))
    return render_template('processing.html')

@app.route('/stream')
def stream():
    """
    This is the endpoint that the processing page connects to. It streams
    the output of the main pipeline generator.
    """
    if not session.get('is_processing'):
        return Response(status=404)
    
    # THE FIX: Read eps from session here, and pass it to the generator.
    eps_value = session.get('eps_value', 0.5)
    
    # The generator function is the source of our SSE stream
    return Response(run_pipeline_and_yield_progress(eps_value=eps_value), mimetype='text/event-stream')

@app.route('/face_thumbnails/<filename>')
def serve_face_thumbnail(filename):
    return send_from_directory(app.config['THUMBNAIL_FOLDER'], filename)

@app.route('/output_images_with_boxes/<filename>')
def serve_processed_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=False)

# --- Main execution for Flask app ---
if __name__ == '__main__':
    # The old script-style main block is replaced by Flask's development server run.
    # Make sure input_images and output_images_with_boxes are at the same level as app.py
    # or adjust app.config['UPLOAD_FOLDER'] and app.config['PROCESSED_FOLDER'] accordingly.
    print(f"Input images expected in: {os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'])}")
    print(f"Processed images will be saved to: {os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['PROCESSED_FOLDER'])}")
    # We disable the reloader to prevent it from restarting the server when we upload files.
    app.run(debug=True, use_reloader=False) # debug=True is helpful for development, auto-reloads on code change.
