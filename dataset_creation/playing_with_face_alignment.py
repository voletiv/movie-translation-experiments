import face_alignment
import time

from movie_translation_data_creation_params import *
from movie_translation_data_creation_functions import *

config = MovieTranslationConfig()

# detector, predictor = load_detector_and_predictor()


# VIDEO FILE!
video_file = '/home/voletiv/Datasets/MOVIE_TRANSLATION/videos/telugu/Mahesh_Babu/Mahesh_Babu_0000.mp4'

# Extract info from video_file's name
video_file_split = video_file.split("/")
video_file_name = os.path.splitext(video_file_split[-1])[0]
actor = video_file_split[-2]
language = video_file_split[-3]
video_frames_dir = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, "frames", language, actor, video_file_name)
video_frames_combined_dir = os.path.join(config.MOVIE_TRANSLATION_DATASET_DIR, "frames_combined", language, actor, video_file_name)

print(language, actor, video_file_name, video_frames_dir, video_frames_combined_dir)

# Extract frames
video_frames = imageio.get_reader(video_file)
frames_list = []
for frame in video_frames:
    frames_list.append(frame)

###################################################################
# MEASURE TIME
###################################################################

# Measure time taken by dlib detector for face detection
start_time = time.time()
for frame in frames_list:
    faces = detector(frame, 1)

end_time = time.time()
print(end_time - start_time)
# 29.19359254837036


# Measure time taken by face_alignment dlib detector for face detection
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=False, flip_input=False, use_cnn_face_detector=False)
start_time = time.time()
for frame in frames_list:
    faces = fa.detect_faces(frame)

end_time = time.time()
print(end_time - start_time)
# 23.053516626358032


# Measure time taken by face_alignment CNN face detector with CUDA for face detection
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=False, use_cnn_face_detector=True)
start_time = time.time()
for frame in frames_list:
    faces = fa.detect_faces(frame)

end_time = time.time()
print(end_time - start_time)
# 64.09909152984619


# Measure time taken by face_alignment CNN face detector WITHOUT CUDA for face detection
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=False, flip_input=False, use_cnn_face_detector=True)
start_time = time.time()
for frame in frames_list:
    faces = fa.detect_faces(frame)

end_time = time.time()
print(end_time - start_time)
# 66.15975999832153


# HENCE, CHOOSING face_alignment dlib detector, trying on landmark detection

# Measure time taken by dlib for 2D landmark detection
start_time = time.time()
for frame in frames_list:
    landmarks = get_landmarks(frame, detector, predictor)

end_time = time.time()
print(end_time - start_time)
# 23.26197028160095

# Measure time taken by face_alignment for 3D landmark detection WITHOUT CUDA
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=False, flip_input=False, use_cnn_face_detector=False)
start_time = time.time()
for frame in frames_list:
    landmarks = fa.get_landmarks(frame)

end_time = time.time()
print(end_time - start_time)
# 137.31819462776184

# Measure time taken by face_alignment for 3D landmark detection WITH CUDA
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=False, use_cnn_face_detector=False)
start_time = time.time()
for frame in frames_list:
    landmarks = fa.get_landmarks(frame)

end_time = time.time()
print(end_time - start_time)
# 75.88490056991577



