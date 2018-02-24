from movie_translation_data_creation_params import *


def load_detector_and_predictor(verbose=False):
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        if verbose:
            print("Detector and Predictor loaded. (load_detector_and_predictor)")
        return detector, predictor
    # If error in SHAPE_PREDICTOR_PATH
    except RuntimeError:
        raise ValueError("\n\nERROR: Wrong Shape Predictor .dat file path - " + \
            SHAPE_PREDICTOR_PATH, "(load_detector_and_predictor)\n\n")


def read_metadata(metadata_txt_file):
    d = []
    with open(metadata_txt_file, 'r') as f:
        for line in f:
            d.append(line.split())
    return d


def extract_video_clips(language, actor, metadata, youtube_videos_dir=YOUTUBE_VIDEOS_DIR, verbose=False):
    # Make video_clips_dir
    video_clips_dir = os.path.join(DATASET_DIR, "videos", language, actor)
    if not os.path.exists(video_clips_dir):
        if verbose:
            print("Making dir", video_clips_dir)
        os.makedirs(video_clips_dir)
    # Read each column of metadata
    # output_video_file_name | youtube_URL | start_time | duration
    output_video_file_names = []
    youtube_URLs = []
    start_times = []
    durations = []
    for line in metadata:
        output_video_file_names.append(line[0])
        youtube_URLs.append(line[1])
        start_times.append(line[2])
        durations.append(line[3])
    # Extract clips
    for output_video_file_name, youtube_URL, start_time, duration in tqdm.tqdm(zip(output_video_file_names, youtube_URLs, start_times, durations), total=len(durations)):
        output_video = os.path.join(video_clips_dir, output_video_file_name)
        input_video = os.path.join(youtube_videos_dir, youtube_URL + '.mp4')
        # ffmpeg -ss 00:08:31 -i LS6XiINMc2s.mp4 -t 00:00:01.5 -y -vcodec libx264 -preset ultrafast -profile:v main -acodec aac -strict -2 newStream1.mp4
        command = ['ffmpeg', '-loglevel', 'warning', '-ss', start_time, '-i', input_video, '-t', duration, '-y',
                   '-vcodec', 'libx264', '-preset', 'ultrafast', '-profile:v', 'main', '-acodec', 'aac', '-strict', '-2', output_video]
        if verbose:
            print(" ".join(command))
        subprocess.call(command)


def extract_face_frames_from_video(video_file, detector, predictor, save_gif=True, save_landmarks_as_txt=True, save_landmarks_as_csv=False, verbose=False):
    '''
    Extract face frames using landmarks, and save in DATASET_DIR/frames/language/actor/video_file
    [optional] Save all face frames as gif
    [optional] Save landmarks in DATASET_DIR/landmarks/language/actor/video_file
    '''
    video_file_split = video_file.split("/")
    video_file_name = os.path.splitext(video_file_split[-1])[0]
    actor = video_file_split[-2]
    language = video_file_split[-3]
    video_frames_dir = os.path.join(DATASET_DIR, "frames", language, actor, video_file_name)
    # Make video_frames_dir
    if not os.path.exists(video_frames_dir):
        os.makedirs(video_frames_dir)
    # Make landmarks_dir
    if save_landmarks_as_txt or save_landmarks_as_csv:
        landmarks_dir = os.path.join(DATASET_DIR, "landmarks", language, actor)
        if not os.path.exists(landmarks_dir):
            os.makedirs(landmarks_dir)
    # Read video
    video_frames = imageio.get_reader(video_file)
    if save_gif:
        faces_list = []
    if save_landmarks_as_txt or save_landmarks_as_csv:
        landmarks_list = []
    for f, frame in tqdm.tqdm(enumerate(video_frames), total=len(video_frames)):
        video_frame_name = video_file_name + "_frame_{0:03d}.png".format(f)
        # Get landmarks
        landmarks = get_landmarks(frame, detector, predictor)
        if landmarks:
            # Crop 1.5x face and resize to 224x224
            face_square_expanded_resized, landmarks_in_face_square_expanded_resized = crop_face_good(frame, landmarks)
            if save_gif:
                faces_list.append(face_square_expanded_resized)
            if save_landmarks_as_txt or save_landmarks_as_csv:
                landmarks_list.append([video_frame_name] + landmarks_in_face_square_expanded_resized)
            # Write face image
            cv2.imwrite(os.path.join(video_frames_dir, video_frame_name), cv2.cvtColor(face_square_expanded_resized, cv2.COLOR_RGB2BGR))
        else:
            if save_landmarks_as_txt or save_landmarks_as_csv:
                landmarks_list.append([video_frame_name] + [])
    # Save gif
    if save_gif:
        imageio.mimsave(os.path.join(video_frames_dir, video_file_name + ".gif"), faces_list)
    # Save landmarks
    # txt is smaller than csv
    if save_landmarks_as_txt:
        write_landmarks_list_as_txt(os.path.join(landmarks_dir, video_file_name + "_landmarks.txt"), landmarks_list)
    if save_landmarks_as_csv:
        write_landmarks_list_as_csv(os.path.join(landmarks_dir, video_file_name + "_landmarks.csv"), landmarks_list)


def get_landmarks(frame, detector, predictor):
    # Landmarks Coords: ------> x (cols)
    #                  |
    #                  |
    #                  v
    #                  y
    #                (rows)
    faces = detector(frame, 1)
    if len(faces) > 0:
        face = faces[0]
        # Detect landmarks
        shape = predictor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), face)
        landmarks = [[shape.part(i).x, shape.part(i).y] for i in range(68)]
        return landmarks
    else:
        return []


def crop_face_good(frame, landmarks):
    # - Get face bounding box from landmarks
    # - Make face bounding box square to the higher of width and height
    # - Expand face bounding box to 1.5x
    # - Resize frame[face_bounding_box] to 224x224
    landmarks_np = np.array(landmarks)
    # dlib.rectangle = left, top, right, bottom
    face_rect = dlib.rectangle(int(np.min(landmarks_np[:, 0])), int(np.min(landmarks_np[:, 1])), int(np.max(landmarks_np[:, 0])), int(np.max(landmarks_np[:, 1])))
    face_rect_square = make_rect_shape_square(face_rect)
    face_rect_square_expanded = expand_rect(face_rect_square, scale=1.5, frame_shape=(frame.shape[0], frame.shape[1]))
    face_square_expanded = frame[face_rect_square_expanded.top():face_rect_square_expanded.bottom(), face_rect_square_expanded.left():face_rect_square_expanded.right()]
    face_square_expanded_resized = np.round(resize(face_square_expanded, (224, 224), preserve_range=True)).astype('uint8')
    landmarks_in_face_square_expanded_resized = [[int((x-face_rect_square_expanded.left())/(face_rect_square_expanded.right() - face_rect_square_expanded.left())*224),
                                                  int((y-face_rect_square_expanded.top())/(face_rect_square_expanded.bottom() - face_rect_square_expanded.top())*224)] for (x, y) in landmarks]
    return face_square_expanded_resized, landmarks_in_face_square_expanded_resized


def make_rect_shape_square(rect):
    # dlib.rectangle
    if type(rect) == dlib.rectangle:
        x = rect.left()
        y = rect.top()
        w = rect.right() - rect.left()
        h = rect.bottom() - rect.top()
    else:
        # Rect: (x, y, w, h)
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
    # If width > height
    if w > h:
        new_x = x
        new_y = int(y + h/2 - w/2)
        new_w = w
        new_h = w
    # Else (height > width)
    else:
        new_x = int(x + w/2 - h/2)
        new_y = y
        new_w = h
        new_h = h
    # Return
    if type(rect) == dlib.rectangle:
        return dlib.rectangle(new_x, new_y, new_x + new_w, new_y + new_h)
    else:
        return [new_x, new_y, new_w, new_h]


def expand_rect(rect, scale=1.5, frame_shape=(256, 256)):
    # dlib.rectangle
    if type(rect) == dlib.rectangle:
        x = rect.left()
        y = rect.top()
        w = rect.right() - rect.left()
        h = rect.bottom() - rect.top()
    else:
        # Rect: (x, y, w, h)
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_x = max(0, min(frame_shape[1] - w, x - int((new_w - w) / 2)))
    new_y = max(0, min(frame_shape[0] - h, y - int((new_h - h) / 2)))
    # w = min(w, frame_shape[1] - x)
    # h = min(h, frame_shape[0] - y)
    if type(rect) == dlib.rectangle:
        return dlib.rectangle(new_x, new_y, new_x + new_w, new_y + new_h)
    else:
        return [new_x, new_y, new_w, new_h]


def write_landmarks_list_as_csv(path, landmarks_list):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(landmarks_list)


def write_landmarks_list_as_txt(path, landmarks_list):
    with open(path, "w") as f:
        for row in landmarks_list:
            line = ""
            for e in row:
                line += str(e) + " "
            line = line[:-1] + "\n"
            f.write(line)


def read_landmarks_list_from_txt(path):
    landmarks_list = []
    translate_table = dict((ord(char), None) for char in '[],')
    with open(path, "r") as f:
        for line in f:
            row = line.strip().split(" [")
            landmarks_list.append([row[0]] + [[int(e.split(" ")[0].translate(translate_table)), int(e.split(" ")[1].translate(translate_table))] for e in row[1:]])
    return landmarks_list


def plot_landmarks(frame, landmarks):
    frame = np.array(frame)
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    plt.imshow(frame)
    plt.show()


def watch_video(video_frames):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    plt.ion()
    fig.show()
    fig.canvas.draw()
    for f, frame in enumerate(video_frames):
        ax.imshow(frame)
        ax.set_title(str(f))
        fig.canvas.draw()


def correct_video_numbers_in_metadata(d, actor, write=False, txt_file_path=None):
    for i, l in enumerate(d):
        l[0] = actor + '_{0:04d}.mp4'.format(i)
    if write:
        if txt_file_path is None:
            txt_file_path = os.path.join(os.getcwd(), actor + '_.txt')
        print("Writing corrected metadata in", )
        with open(txt_file_path, 'w') as f:
            for l in d:
                f.write(" ".join(l) + "\n")
    return d
