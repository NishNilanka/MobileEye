import csv
import os
import sys
import time

import cv2
import dlib
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.set_visible_devices(physical_devices[0], 'GPU')

detector = dlib.get_frontal_face_detector()


def get_dlib_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)
    if len(faces) == 0:
        return None

    # Loop over the detected faces
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Crop the detected face from the original frame
        face_crop = frame[y:y + h, x:x + w]

    # cv2.imshow('Frame', gray)
    return face_crop

def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


# Load the Haar Cascade classifier for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


def detect_face(frame):
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face on the original frame
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = gray[y:y + h, x:x + w]

    # print(type(cropped_face))
    return cropped_face


def resize_frame(frame):
    return cv2.resize(frame, (128, 128))


def preprocess_frame(frame, target_dim):
    frame = resize_frame(frame)
    frame = tf.convert_to_tensor(frame, dtype=tf.float32)
    shape = tf.cast(tf.shape(frame)[0:2], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    frame = tf.expand_dims(frame, axis=0)
    frame = tf.expand_dims(frame, axis=-1)

    frame = tf.image.resize(frame, new_shape)
    frame = tf.image.resize_with_crop_or_pad(frame, target_dim, target_dim)
    return frame


def preprocess_image(image, target_dim):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (target_dim, target_dim))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image[tf.newaxis, :]
    return image



def write_to_CSV(coordinates_array, filename):
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(["sid", "frameid", "pred_x", "pred_y"])

        for entry in coordinates_array:
            video = entry[0]
            frame_count = entry[1]
            coordinates = entry[2]
            csvwriter.writerow([video, frame_count, *coordinates])


def process_videos(videos_directory, model, target_dim, csv_file):
    total_execution_time = 0.0
    frame_count = 0
    total_execution_time_face = 0.0
    total_execution_time_read = 0.0
    total_execution_time_model = 0.0
    total_execution_time_preprocessing = 0.0
    video_files = os.listdir(videos_directory)

    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["sid", "frameid", "pred_x", "pred_y"])

        for video_file in video_files:
            video_path = os.path.join(videos_directory, video_file)
            video = video_file.split(" ")[0]
            # Load the video
            cap = cv2.VideoCapture(video_path)

            predicted_coordinates = []

            while True:
                # Measure the execution time for each frame
                start_time = time.time()
                start_time_read = time.time()
                ret, frame = cap.read()

                # Break the loop if the video has ended
                if not ret:
                    break

                execution_time_read = (time.time() - start_time_read) * 1000.0
                total_execution_time_read += execution_time_read

                start_time_face = time.time()
                #face = detect_face(frame)

                face = get_dlib_face(frame)


                execution_time_face = (time.time() - start_time_face) * 1000.0
                total_execution_time_face += execution_time_face
                if face is None:
                    continue

                start_time_preprocessing = time.time()

                # frame_tensor  = tf.convert_to_tensor(face, dtype=tf.float32)
                resized = preprocess_image(face , 128)
                # resized = preprocess_frame(face, target_dim)
                execution_time_preprocessing = (time.time() - start_time_preprocessing) * 1000.0
                total_execution_time_preprocessing += execution_time_preprocessing

                start_time_model = time.time()
                # print(tf.shape(preprocessed_frame))
                # predicted_coordinates.append([video, frame_count, model.predict(preprocessed_frame, verbose=0)])

                predicted_coordinates.append([video, frame_count, *model.predict(resized , verbose=0)])
                execution_time_model = (time.time() - start_time_model) * 1000.0
                total_execution_time_model += execution_time_model

                execution_time = (time.time() - start_time) * 1000.0
                total_execution_time += execution_time
                # Display the frame with detected faces
                # cv2.imshow('Frame', face)
                # Exit the loop if 'q' is pressed

                # csvwriter.writerow([video, frame_count, *model.predict(resized, verbose=0)])
                predictions = model.predict(resized, verbose=0)
                formatted_predictions = [float(predictions[0][0]), float(predictions[0][1])]
                csvwriter.writerow([video, frame_count, f"{formatted_predictions[0]:.6f},{formatted_predictions[1]:.6f}"])

                frame_count += 1

            cap.release()
            cv2.destroyAllWindows()

    average_execution_time = total_execution_time / frame_count
    average_time_face = total_execution_time_face / frame_count
    average_time_read = total_execution_time_read / frame_count
    average_time_model = total_execution_time_model / frame_count
    average_time_preprocessing = total_execution_time_preprocessing / frame_count


    # write_to_CSV(predicted_coordinates, csv_file)

    # call the extract_column function

    return average_execution_time, average_time_face, average_time_read, average_time_model, average_time_preprocessing, frame_count




def run_process(model_path):
    # video_path = '../videos/'
    video_path = '../one_video/'

    # model_paths = ['models/CNN_2D_RANDOM', 'models/CNN_iTracker_RANDOM', 'models/CNN_LSTM_RANDOM', 'models/CNN_GRU_RANDOM']

    dst_file = model_path.split('/')[1]+'.csv'

    new_model = tf.keras.models.load_model(model_path)
    average_time, average_time_face, average_time_read, average_time_model, average_time_preprocessing, frame_count = process_videos(
        video_path, new_model, 128, dst_file)

    print(f"Model: {model_path}")
    print(f"Average Execution time: {average_time:.2f} milliseconds")
    print(f"Frame Count for: {frame_count}")

    print(f"Average Execution time [Frame Read]: {average_time_read:.2f} milliseconds")
    print(f"Average Execution time [Face Detection]: {average_time_face:.2f} milliseconds")
    print(f"Average Execution time [Pre-Processing]: {average_time_preprocessing:.2f} milliseconds")
    print(f"Average Execution time [Model Inference]: {average_time_model:.2f} milliseconds")

    txt_file = 'Results_' + model_path.split('/')[1]+'.txt'
    with open(txt_file, 'w') as file:
        # Write average execution times and frame count to the file
        file.write(f"Model: {model_path}\n")
        file.write(f"Average Execution time: {average_time:.2f} milliseconds\n")
        file.write(f"Frame Count for: {frame_count}\n")
        file.write(f"Average Execution time [Frame Read]: {average_time_read:.2f} milliseconds\n")
        file.write(f"Average Execution time [Face Detection]: {average_time_face:.2f} milliseconds\n")
        file.write(f"Average Execution time [Pre-Processing]: {average_time_preprocessing:.2f} milliseconds\n")
        file.write(f"Average Execution time [Model Inference]: {average_time_model:.2f} milliseconds\n")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    run_process(model_path)
