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
        video1 = os.path.join(youtube_videos_dir, youtube_URL + '.mp4')
        # ffmpeg -ss 00:08:31 -i LS6XiINMc2s.mp4 -t 00:00:01.5 -y -vcodec libx264 -preset ultrafast -profile:v main -acodec aac -strict -2 newStream1.mp4
        command = ['ffmpeg', '-loglevel', 'warning', '-ss', start_time, '-i', video1, '-t', duration, '-y',
                   '-vcodec', 'libx264', '-preset', 'ultrafast', '-profile:v', 'main', '-acodec', 'aac', '-strict', '-2', output_video]
        if verbose:
            print(" ".join(command))
        subprocess.call(command)


def extract_face_frames_from_video(video_file, detector, predictor, save_with_blackened_mouths_and_polygons=True, save_gif=True, save_landmarks_as_txt=True, save_landmarks_as_csv=False, verbose=False):
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
    for frame_number, frame in tqdm.tqdm(enumerate(video_frames), total=len(video_frames)):
        video_frame_name = video_file_name + "_frame_{0:03d}.png".format(frame_number)
        # Get landmarks
        landmarks = get_landmarks(frame, detector, predictor)
        if landmarks:
            # Crop 1.5x face and resize to 224x224
            face_square_expanded_resized, landmarks_in_face_square_expanded_resized = crop_face_good(frame, landmarks)
            if save_gif:
                faces_list.append(face_square_expanded_resized)
            if save_landmarks_as_txt or save_landmarks_as_csv:
                landmarks_list.append([video_frame_name] + landmarks_in_face_square_expanded_resized)
            if save_with_blackened_mouths_and_polygons:
                # Make new frame with blackened mouth
                mouth_landmarks = np.array(landmarks_in_face_square_expanded_resized[48:68])
                mouth_rect = dlib.rectangle(int(np.min(mouth_landmarks[:, 0])), int(np.min(mouth_landmarks[:, 1])), int(np.max(mouth_landmarks[:, 0])), int(np.max(mouth_landmarks[:, 1])))
                mouth_rect_expanded = expand_rect(mouth_rect, scale_w=1.2, scale_h=1.8, frame_shape=(224, 224))
                # Blacken mouth in frame
                frame_with_blackened_mouth = np.array(frame)
                frame_with_blackened_mouth[mouth_rect_expanded.top():mouth_rect_expanded.bottom(), mouth_rect_expanded.left():mouth_rect_expanded.right()] = 0
                # Draw mouth polygon in frame
                frame_with_blackened_mouth_and_mouth_polygon = np.array(frame_with_blackened_mouth)
                frame_with_blackened_mouth_and_mouth_polygon = cv2.drawContours(frame_with_blackened_mouth_and_mouth_polygon, [mouth_landmarks[:12], mouth_landmarks[12:]], -1, (255, 255, 255))
                # Write combined frame+frame_with_blacked_mouth_and_polygon image
                frame_combined = np.hstack((frame, frame_with_blackened_mouth_and_mouth_polygon))
                cv2.imwrite(os.path.join(video_frames_dir, video_frame_name), cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR))
            else:
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


def expand_rect(rect, scale=None, scale_w=1.5, scale_h=1.5, frame_shape=(256, 256)):
    if scale is not None:
        scale_w = scale
        scale_h = scale
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
    new_w = int(w * scale_w)
    new_h = int(h * scale_h)
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


def make_blackened_mouths_and_mouth_polygons(video_name):
    # Read landmarks
    landmarks_file = os.path.join(DATASET_DIR, 'landmarks', language, actor, video_name + "_landmarks.txt")
    video_landmarks = read_landmarks_list_from_txt(landmarks_file)
    video_frames_dir = os.path.join(DATASET_DIR, 'frames', language, actor, video_name)
    # Folders
    video_frames_and_black_mouths_combined_dir = os.path.join(DATASET_DIR, 'frames_combined', language, actor, video_name)
    if not os.path.exists(video_frames_and_black_mouths_combined_dir):
        os.makedirs(video_frames_and_black_mouths_combined_dir)
    # For each frame
    for frame_number, frame_name_and_landmarks in enumerate(video_landmarks):
        frame_name = frame_name_and_landmarks[0]
        frame_landmarks = frame_name_and_landmarks[1:]
        if len(frame_name_and_landmarks) == 69:
            frame = cv2.cvtColor(cv2.imread(os.path.join(video_frames_dir, frame_name)), cv2.COLOR_BGR2RGB)
            mouth_landmarks = np.array(frame_landmarks[48:68])
            mouth_rect = dlib.rectangle(int(np.min(mouth_landmarks[:, 0])), int(np.min(mouth_landmarks[:, 1])), int(np.max(mouth_landmarks[:, 0])), int(np.max(mouth_landmarks[:, 1])))
            mouth_rect_expanded = expand_rect(mouth_rect, scale_w=1.2, scale_h=1.8, frame_shape=(224, 224))
            # Blacken mouth in frame
            frame_with_blackened_mouth = np.array(frame)
            frame_with_blackened_mouth[mouth_rect_expanded.top():mouth_rect_expanded.bottom(), mouth_rect_expanded.left():mouth_rect_expanded.right()] = 0
            # Draw mouth polygon in frame
            frame_with_blackened_mouth_and_mouth_polygon = np.array(frame_with_blackened_mouth)
            frame_with_blackened_mouth_and_mouth_polygon = cv2.drawContours(frame_with_blackened_mouth_and_mouth_polygon, [mouth_landmarks[:12], mouth_landmarks[12:]], -1, (255, 255, 255))
            # Write image
            frame_combined = np.hstack((frame, frame_with_blackened_mouth_and_mouth_polygon))
            cv2.imwrite(os.path.join(video_frames_and_black_mouths_combined_dir, video_name + "_frame_combined_{0:03d}.png".format(frame_number)), cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR))


def read_log_and_plot_graphs(log_txt_path):
    log_lines = []
    with open(log_txt_path, 'r') as f:
        for line in f:
            log_lines.append(line)
    D_log_losses = []
    G_tot_losses = []
    G_l1_losses = []
    G_log_losses = []
    for line in log_lines:
        if 'D logloss' in line:
            line_split = line.split()
            D_log_losses.append(float(line_split[8]))
            G_tot_losses.append(float(line_split[12]))
            G_l1_losses.append(float(line_split[16]))
            G_log_losses.append(float(line_split[20][:6]))
    plt.figure()
    plt.subplot(121)
    plt.plot(np.arange(len(D_log_losses)), D_log_losses)
    plt.xlabel("Epochs")
    plt.title("Discriminator loss")
    plt.subplot(122)
    plt.plot(np.arange(len(G_tot_losses)), G_tot_losses, linewidth=2, label='G_total_loss')
    plt.plot(np.arange(len(G_l1_losses)), G_l1_losses, label='G_L1_loss')
    plt.plot(np.arange(len(G_log_losses)), G_log_losses, label='G_log_loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.title("Generator loss")
    plt.show()


def plot_lip_landmarks(lip_landmarks, video=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if video:
        plt.ion()
        fig.show()
        fig.canvas.draw()
    a = np.zeros((224, 224))
    for l, lip_landmark in enumerate(lip_landmarks):
        a[lip_landmark[1]-3:lip_landmark[1]+3, lip_landmark[0]-3:lip_landmark[0]+3] = 1
        ax.imshow(a)
        if video:
            ax.set_title(str(l))
            fig.canvas.draw()
    if not video:
        plt.show()
