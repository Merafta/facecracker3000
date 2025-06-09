import os
import glob
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from PIL import Image, ImageDraw
from flask import Flask, render_template, url_for, send_from_directory, request, redirect, flash, Response, jsonify
import zipfile
import shutil
import time
import logging

# --- App Constants ---
FACE_DETECTION_MODEL = "hog" # "hog" or "cnn"

# --- Flask App Setup ---
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
# Add a secret key for flashing messages
app.secret_key = 'supersecretkey' # In production, this should be a real secret
# Configuration for folders
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output_images_with_boxes'
app.config['INPUT_FOLDER'] = 'input_images'
app.config['FACE_THUMBNAILS_FOLDER'] = 'face_thumbnails' # New config

# Global variable to store processed results to avoid reprocessing on every request (for POC)
processed_data_for_web = None
# Ensure all directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['INPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['FACE_THUMBNAILS_FOLDER'], exist_ok=True) # New directory creation

@app.before_request
def log_request_info():
    """Log information about the incoming request before handling it."""
    # This will print to the terminal for every request the server receives.
    print(f"--- [DEBUG] Incoming Request: {request.method} {request.path} ---")

# --- New Helper Function to Clear Data ---
def clear_all_data():
    """Clears previously uploaded and processed files."""
    global processed_data_for_web
    processed_data_for_web = None # Reset the cache

    print("Clearing previous data...")
    for folder_key in ['INPUT_FOLDER', 'OUTPUT_FOLDER', 'UPLOAD_FOLDER', 'FACE_THUMBNAILS_FOLDER']: # Add new folder to cleanup
        folder_path = app.config[folder_key]
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
    print("Data cleared.")

def get_image_paths(folder_path, image_extensions=None):
    """
    Scans a folder for image files and returns a list of their paths.

    Args:
        folder_path (str): The path to the folder containing images.
        image_extensions (list, optional): A list of image extensions to look for.
                                           Defaults to ['.jpg', '.jpeg', '.png'].

    Returns:
        list: A list of absolute paths to the found image files.
              Returns an empty list if the folder doesn't exist or no images are found.
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png']

    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return []

    image_paths = []
    for ext in image_extensions:
        # Using os.path.join for cross-platform compatibility
        # Using glob.glob to find files matching the pattern
        # Adding '*' to match any filename with the given extension
        search_pattern = os.path.join(folder_path, f"*{ext.lower()}")
        image_paths.extend(glob.glob(search_pattern))
        # Also search for uppercase extensions, just in case
        search_pattern_upper = os.path.join(folder_path, f"*{ext.upper()}")
        image_paths.extend(glob.glob(search_pattern_upper))
    
    # Remove duplicates that might arise from case-insensitive filesystems
    # and searching for both lower and upper case extensions
    return sorted(list(set(image_paths)))

def cluster_faces(face_data, eps=0.43, min_samples=2):
    """
    Clusters face encodings using DBSCAN to identify unique individuals.

    Args:
        face_data (list): A list of dictionaries, where each dictionary contains
                          at least an 'encoding' key with a face encoding.
        eps (float): The maximum distance between samples for one to be
                     considered as in the neighborhood of the other (DBSCAN param).
        min_samples (int): The number of samples in a neighborhood for a point
                           to be considered as a core point (DBSCAN param).

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

    print(f"Clustering {len(encodings)} face encodings using DBSCAN with eps={eps}, min_samples={min_samples}...")

    # Initialize DBSCAN
    # The default metric is 'euclidean', which is suitable for face_recognition encodings.
    clt = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
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

def draw_boxes_and_save(pil_img, face_locations_for_this_image, original_image_path, base_output_folder):
    """
    Draws bounding boxes on a given Pillow image object and saves it.
    Args:
        pil_img (PIL.Image): The Pillow image object (potentially resized).
        face_locations_for_this_image (list): List of (top, right, bottom, left) tuples for faces.
        original_image_path (str): Path to the original image (used for naming).
        base_output_folder (str): The absolute path to the folder where the processed image will be saved.
    Returns:
        str: The filename of the saved image with boxes (not the full path). None if error.
    """
    try:
        # We no longer need to open the image, it's passed directly
        img = pil_img.copy() # Work on a copy to avoid modifying the original object
            
        draw = ImageDraw.Draw(img)

        valid_faces_drawn = 0
        for i, (top, right, bottom, left) in enumerate(face_locations_for_this_image):
            # Add a check for valid coordinates
            if top >= bottom or left >= right:
                print(f"    Warning: Invalid bounding box coordinates for face {i+1} in {original_image_path}: (T:{top}, R:{right}, B:{bottom}, L:{left}). Skipping this box.")
                continue
            draw.rectangle(((left, top), (right, bottom)), outline="red", width=5)
            valid_faces_drawn += 1

        if not valid_faces_drawn and face_locations_for_this_image: # Some locations existed but none were valid
            print(f"    Warning: No valid faces could be drawn for {original_image_path} despite {len(face_locations_for_this_image)} detections.")
            # Decide if you still want to save the image without boxes or return None
            # For now, we will save it without boxes if any attempt to draw was made.
            # If no valid faces drawn, but we intended to draw, perhaps it shouldn't be considered 'boxed'
            # However, the current logic saves it anyway. This could be refined.

        original_filename = os.path.basename(original_image_path)
        output_filename = f"boxed_{original_filename}"
        output_path = os.path.join(base_output_folder, output_filename)
        
        img.save(output_path)
        return output_filename
    except Exception as e:
        print(f"Error drawing boxes for {original_image_path}: {e}")
        return None

def crop_save_and_get_best_thumbnail(faces_for_person, person_name, thumbnails_folder):
    """Finds the best face, crops it, saves it, and returns the filename."""
    if not faces_for_person:
        return None

    best_face = None
    max_area = -1

    # Find the face with the largest bounding box area
    for face in faces_for_person:
        top, right, bottom, left = face['location']
        area = (bottom - top) * (right - left)
        if area > max_area:
            max_area = area
            best_face = face

    if best_face is None:
        return None

    try:
        # Crop the face from the resized image object
        image_to_crop = best_face['resized_image_obj']
        top, right, bottom, left = best_face['location']
        cropped_face = image_to_crop.crop((left, top, right, bottom))

        # Save the thumbnail
        thumbnail_filename = f"thumb_{person_name}.jpg".replace(" ", "_")
        thumbnail_path = os.path.join(thumbnails_folder, thumbnail_filename)
        cropped_face.save(thumbnail_path, "JPEG")
        
        return thumbnail_filename
    except Exception as e:
        print(f"Error creating thumbnail for {person_name}: {e}")
        return None

def prepare_web_output(face_data, cluster_labels, abs_output_folder):
    """
    Processes face data and cluster labels to draw bounding boxes and group images.
    Args:
        face_data (list): List of dictionaries from detect_and_encode_faces.
        cluster_labels (numpy.ndarray): Cluster labels from cluster_faces.
        abs_output_folder (str): Absolute path to the folder to save images with boxes.

    Returns:
        tuple: (people_to_processed_images, unclustered_processed_images)
               people_to_processed_images (dict): {person_id_str: [{'original_filename': str, 'processed_filename': str}]}
               unclustered_processed_images (list): [{'original_filename': str, 'processed_filename': str}]
    """
    if cluster_labels is None or (face_data and len(face_data) != len(cluster_labels)):
        print("Error: Face data and cluster labels mismatch or labels are missing for web prep.")
        return {}, []

    # Step 1: Group all face data by cluster label
    # This helps in finding the best thumbnail for each person
    faces_by_person_label = defaultdict(list)
    if face_data:
        for i, data_point in enumerate(face_data):
            label = cluster_labels[i]
            if label != -1:
                faces_by_person_label[label].append(data_point)

    # Step 2: Draw boxes on unique images (this part remains the same)
    image_path_to_boxed_filename = {}
    unique_images_and_their_faces = defaultdict(lambda: {'locations': [], 'image_obj': None})
    if face_data:
        for data_point in face_data:
            path = data_point['image_path']
            unique_images_and_their_faces[path]['locations'].append(data_point['location'])
            if unique_images_and_their_faces[path]['image_obj'] is None:
                unique_images_and_their_faces[path]['image_obj'] = data_point['resized_image_obj']

    for original_path, data in unique_images_and_their_faces.items():
        if data['locations'] and data['image_obj']:
            boxed_filename = draw_boxes_and_save(data['image_obj'], data['locations'], original_path, abs_output_folder)
            if boxed_filename:
                image_path_to_boxed_filename[original_path] = boxed_filename

    # Step 3: Build the final data structure for the web page
    people_to_display_data = defaultdict(lambda: {'images': [], 'thumbnail_filename': None})
    unclustered_display_data = []
    
    person_id_counter = 1
    unique_cluster_labels_sorted = sorted(list(set(l for l in cluster_labels if l != -1)))
    label_to_person_name = {}
    for label_val in unique_cluster_labels_sorted:
        person_name = f"Person_{person_id_counter}"
        label_to_person_name[label_val] = person_name
        person_id_counter += 1

        # Create thumbnail for this person
        thumbnail_filename = crop_save_and_get_best_thumbnail(
            faces_by_person_label[label_val],
            person_name,
            app.config['FACE_THUMBNAILS_FOLDER']
        )
        if thumbnail_filename:
            people_to_display_data[person_name]['thumbnail_filename'] = thumbnail_filename
            
    person_images_added_to_web = defaultdict(set)
    unclustered_images_added_to_web = set()

    if face_data:
        for i, data_point in enumerate(face_data):
            original_path = data_point['image_path']
            original_basename = os.path.basename(original_path)
            boxed_filename = image_path_to_boxed_filename.get(original_path)
            if not boxed_filename:
                continue 

            image_display_info = {'original_filename': original_basename, 'processed_filename': boxed_filename}
            current_label = cluster_labels[i]

            if current_label != -1:
                person_name = label_to_person_name[current_label]
                if boxed_filename not in person_images_added_to_web[person_name]:
                    people_to_display_data[person_name]['images'].append(image_display_info)
                    person_images_added_to_web[person_name].add(boxed_filename)
            else:
                if boxed_filename not in unclustered_images_added_to_web:
                    unclustered_display_data.append(image_display_info)
                    unclustered_images_added_to_web.add(boxed_filename)
    
    for person_name in people_to_display_data:
        people_to_display_data[person_name]['images'].sort(key=lambda x: x['original_filename'])
    unclustered_display_data.sort(key=lambda x: x['original_filename'])

    return people_to_display_data, unclustered_display_data

# --- New Generator Function for the Processing Pipeline ---
def run_pipeline_and_yield_progress():
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
        input_dir_abs_path = os.path.join(base_dir, app.config['INPUT_FOLDER'])
        output_dir_abs_path = os.path.join(base_dir, app.config['OUTPUT_FOLDER'])

        yield sse_message({"message": "Scanning for images..."})
        image_files = get_image_paths(input_dir_abs_path)
        if not image_files:
            yield sse_message({"message": "Error: No valid image files found in the zip.", "error": True})
            return
        
        total_images = len(image_files)
        yield sse_message({"message": f"Found {total_images} images. Starting face detection..."})
        
        # --- Refactor detect_and_encode_faces to yield progress ---
        all_face_data = []
        total_faces_found = 0
        MAX_IMAGE_DIMENSION = 1600 # New constant for resizing

        for i, image_path in enumerate(image_files):
            progress_percent = int(((i + 1) / total_images) * 100)
            msg = f"({i+1}/{total_images}) Processing {os.path.basename(image_path)}..."
            yield sse_message({"message": msg, "progress": progress_percent})

            try:
                # --- New Resizing Logic ---
                img_to_process = Image.open(image_path)
                # Ensure image has a valid mode for face_recognition
                if img_to_process.mode not in ['RGB', 'L']:
                    img_to_process = img_to_process.convert('RGB')
                
                # Check if resizing is needed
                if max(img_to_process.size) > MAX_IMAGE_DIMENSION:
                    yield sse_message({
                        "message": f"{msg} Image is large, resizing...",
                        "progress": progress_percent
                    })
                    img_to_process.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)

                # Convert PIL image to numpy array for face_recognition
                image = np.array(img_to_process)
                # --- End of Resizing Logic ---

                # Use the resized 'image' numpy array and the constant for the model
                face_locations = face_recognition.face_locations(image, model=FACE_DETECTION_MODEL)
                
                if face_locations:
                    total_faces_found += len(face_locations)
                    msg_faces = f"Found {len(face_locations)} face(s). Total so far: {total_faces_found}."
                    yield sse_message({"message": f"{msg} {msg_faces}", "progress": progress_percent})
                    face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
                    for loc, enc in zip(face_locations, face_encodings):
                        # Store the resized image object itself along with other data
                        all_face_data.append({
                            'image_path': image_path,
                            'location': loc,
                            'encoding': enc,
                            'resized_image_obj': img_to_process # Add this
                        })
            except Exception as e:
                yield sse_message({"message": f"Error processing {os.path.basename(image_path)}: {e}"})
                continue
        
        yield sse_message({"message": f"Finished image processing. Found {total_faces_found} total faces. Now clustering..."})

        if not all_face_data:
            yield sse_message({"message": "No faces were detected in any images.", "error": True})
            return

        cluster_labels_arr = cluster_faces(all_face_data)
        yield sse_message({"message": "Clustering complete. Generating final output images..."})
        
        people_map, unclustered_list = prepare_web_output(all_face_data, cluster_labels_arr, output_dir_abs_path)
        
        processed_data_for_web = {"people": people_map, "unclustered": unclustered_list, "ran_processing": True}
        yield sse_message({"message": "Processing complete!", "finished": True})

    except Exception as e:
        print(f"Error during pipeline: {e}")
        yield sse_message({"message": f"A critical error occurred: {e}", "error": True})

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # This route now ONLY displays the page with cached data.
    # Pass the model name to the template
    return render_template('index.html', data=processed_data_for_web, model_name=FACE_DETECTION_MODEL)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'zip_file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['zip_file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading."}), 400

    if file and file.filename.endswith('.zip'):
        clear_all_data()
        filename = "upload.zip"
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(zip_path)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(app.config['INPUT_FOLDER'])
            os.remove(zip_path)
            return jsonify({"success": True, "message": "File uploaded and extracted."})
        except Exception as e:
            return jsonify({"error": f"An error occurred during extraction: {e}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a .zip file."}), 400

@app.route('/stream_processing')
def stream_processing():
    """The SSE endpoint that streams progress."""
    return Response(run_pipeline_and_yield_progress(), mimetype='text/event-stream')

@app.route('/face_thumbnails/<filename>')
def serve_face_thumbnail(filename):
    return send_from_directory(app.config['FACE_THUMBNAILS_FOLDER'], filename)

@app.route('/output_images_with_boxes/<filename>')
def serve_processed_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=False)

# --- Main execution for Flask app ---
if __name__ == '__main__':
    # The old script-style main block is replaced by Flask's development server run.
    # Make sure input_images and output_images_with_boxes are at the same level as app.py
    # or adjust app.config['INPUT_FOLDER'] and app.config['OUTPUT_FOLDER'] accordingly.
    print(f"Input images expected in: {os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['INPUT_FOLDER'])}")
    print(f"Processed images will be saved to: {os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['OUTPUT_FOLDER'])}")
    # We disable the reloader to prevent it from restarting the server when we upload files.
    app.run(debug=True, use_reloader=False) # debug=True is helpful for development, auto-reloads on code change.
