from exchange_dialogues_params import *


def load_generator(model_path):
    from keras.models import load_model
    return load_model(model_path)


def exchange_dialogues(generator_model,
                       video1_language="telugu", video1_actor="Mahesh_Babu", video1_number=47,
                       video2_language="telugu", video2_actor="Mahesh_Babu", video2_number=89):
    # Video 1
    video1_frames_dir, video1_landmarks = get_video_frames_dir_and_landmarks(video1_language, video1_actor, video1_number)
    video1_length = len(video1_landmarks)
    # Video 2
    video2_frames_dir, video2_landmarks = get_video_frames_dir_and_landmarks(video2_language, video2_actor, video2_number)
    video2_length = len(video2_landmarks)
    # Choose the smaller one as the target length, and choose those many central frames
    if video1_length < video2_length:
        video_length = video1_length
        video1_frame_numbers = np.arange(video1_length)
        video2_landmarks = video2_landmarks[(video2_length//2 - video1_length//2):(video2_length//2 - video1_length//2 + video1_length)]
        video2_frame_numbers = np.arange((video2_length//2 - video1_length//2), (video2_length//2 - video1_length//2 + video1_length))
    else:
        video_length = video2_length
        video1_landmarks = video1_landmarks[(video1_length//2 - video2_length//2):(video1_length//2 - video2_length//2 + video2_length)]
        video1_frame_numbers = np.arange((video1_length//2 - video2_length//2), (video1_length//2 - video2_length//2 + video2_length))
        video2_frame_numbers = np.arange(video2_length)
    # EXCHANGE DIALOGUES
    new_video1_frames_with_black_mouth_and_lip_polygons = []
    new_video2_frames_with_black_mouth_and_lip_polygons = []
    # For each frame
    # read frame, blacken mouth, make new landmarks' polygon
    for i in range(len(video_length)):

        # Read video1 frame
        video1_frame_name = video1_actor + '_%04d_frame_%03d.png' % (video1_number, video1_frame_numbers[i])
        video1_frame = cv2.cvtColor(cv2.imread(os.path.join(video1_frames_dir, video1_frame_name)), cv2.COLOR_BGR2RGB)

        # Read video2 frame
        video2_frame_name = video2_actor + '_%04d_frame_%03d.png' % (video2_number, video2_frame_numbers[i])
        video2_frame = cv2.cvtColor(cv2.imread(os.path.join(video2_frames_dir, video2_frame_name)), cv2.COLOR_BGR2RGB)

        # Get the landmarks
        video1_frame_lip_landmarks = np.array(video1_landmarks[video1_frame_numbers[i]][1:][48:68])
        video2_frame_lip_landmarks = np.array(video2_landmarks[video2_frame_numbers[i]][1:][48:68])

        # Exchange landmarks
        new_video1_frame_lip_landmarks, new_video2_frame_lip_landmarks = exchange_landmarks(video1_frame_lip_landmarks, video2_frame_lip_landmarks)

        # Make frames with black mouth and polygon of landmarks
        new_video1_frames_with_black_mouth_and_lip_polygons.append(make_black_mouth_and_lips_polygons(video1_frame, new_video1_frame_lip_landmarks))
        new_video2_frames_with_black_mouth_and_lip_polygons.append(make_black_mouth_and_lips_polygons(video2_frame, new_video2_frame_lip_landmarks))

    # Generate new frames
    new_video1_frames_generated = generator_model.predict(new_video1_frames_with_black_mouth_and_lip_polygons)
    new_video2_frames_generated = generator_model.predict(new_video2_frames_with_black_mouth_and_lip_polygons)

    # Save
    imageio.mimsave(os.path.join("video1.gif"), new_video1_frames_generated)
    imageio.mimsave(os.path.join("video2.gif"), new_video2_frames_generated)


#################################################
# DEPENDENT FUNCTIONS
#################################################


def get_video_frames_dir_and_landmarks(language, actor, number):
    frames_dir = os.path.join(DATASET_DIR, 'frames', language, actor, actor + '_%04d' % number)
    landmarks_file = os.path.join(DATASET_DIR, 'landmarks', language, actor, actor + '_%04d' % number + "_landmarks.txt")
    landmarks = read_landmarks_list_from_txt(landmarks_file)
    return frames_dir, landmarks


def exchange_landmarks(video1_frame_lip_landmarks, video2_frame_lip_landmarks):

    # Unrotate both frames' lip landmarks
    video1_frame_lip_landmarks_rotated, angle_video1_landmarks_rotated_by = unrotate_lip_landmarks(video1_frame_lip_landmarks)
    video2_frame_lip_landmarks_rotated, angle_video2_landmarks_rotated_by = unrotate_lip_landmarks(video2_frame_lip_landmarks)

    # Normalize both frames' rotated lip landmarks
    video1_frame_lip_landmarks_rotated_normalized, video1_ur, video1_uc, video1_sr, video1_sc = normalize_lip_landmarks(video1_frame_lip_landmarks_rotated)
    video2_frame_lip_landmarks_rotated_normalized, video2_ur, video2_uc, video2_sr, video2_sc = normalize_lip_landmarks(video2_frame_lip_landmarks_rotated)

    # Make new lip landmarks by unnormalizing and then rotating
    new_video1_frame_lip_landmarks = np.round(rotate_points(unnormalize_lip_landmarks(video2_frame_lip_landmarks_rotated_normalized,
                                                                                      video1_ur, video1_uc, video1_sr, video1_sc),
                                                            angle_video1_landmarks_rotated_by)).astype('int')
    new_video2_frame_lip_landmarks = np.round(rotate_points(unnormalize_lip_landmarks(video1_frame_lip_landmarks_rotated_normalized,
                                                                                      video2_ur, video2_uc, video2_sr, video2_sc),
                                                            angle_video2_landmarks_rotated_by)).astype('int')

    return new_video1_frame_lip_landmarks, new_video2_frame_lip_landmarks


def unrotate_lip_landmarks(lip_landmarks):
    # lip_landmarks = list(lip_landmarks)
    angle_rotated_by = math.atan((lip_landmarks[6][1] - lip_landmarks[0][1])/(lip_landmarks[6][0] - lip_landmarks[0][0]))
    rotated_lip_landmarks = rotate_points(lip_landmarks, lip_landmarks[0], -angle_rotated_by)
    return rotated_lip_landmarks, angle_rotated_by


def rotate_points(points, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    # When the points are row matrices, R is:
    R = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    return origin + np.dot(points-origin, R)


def normalize_lip_landmarks(lip_landmarks):
    ur, uc = lip_landmarks[0]
    sr, sc = lip_landmarks[:, 0].max() - lip_landmarks[:, 0].min(), lip_landmarks[:, 1].max() - lip_landmarks[:, 1].min()
    return (lip_landmarks - [ur, uc])/[sr, sc], ur, uc, sr, sc


def unnormalize_lip_landmarks(lip_landmarks, ur, uc, sr, sc):
    return lip_landmarks * [sr, sc] + [ur, uc]


def make_black_mouth_and_lips_polygons(frame, lip_landmarks):

        # Find mouth bounding box
        mouth_rect = dlib.rectangle(int(np.min(lip_landmarks[:, 0])), int(np.min(lip_landmarks[:, 1])), int(np.max(lip_landmarks[:, 0])), int(np.max(lip_landmarks[:, 1])))

        # Expand mouth bounding box
        mouth_rect_expanded = expand_rect(mouth_rect, scale_w=1.2, scale_h=1.8, frame_shape=(224, 224))

        # Make new frame for blackened mouth and lip polygons
        frame_with_blackened_mouth_and_lip_polygons = np.array(frame)

        # Blacken (expanded) mouth in frame
        frame_with_blackened_mouth_and_lip_polygons[mouth_rect_expanded.top():mouth_rect_expanded.bottom(),
                                                    mouth_rect_expanded.left():mouth_rect_expanded.right()] = 0

        # Draw lips polygon in frame
        frame_with_blackened_mouth_and_lip_polygons = cv2.drawContours(frame_with_blackened_mouth_and_lip_polygons,
                                                                       [lip_landmarks[:12], lip_landmarks[12:]], -1, (255, 255, 255))

        return frame_with_blackened_mouth_and_lip_polygons


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


def unrotate_lip_landmarks_point_by_point(lip_landmarks):
    lip_landmarks = list(lip_landmarks)
    angle_to_rotate_by = -math.atan((lip_landmarks[6][1] - lip_landmarks[0][1])/((224-lip_landmarks[6][0]) - (224-lip_landmarks[0][0])))
    rotated_lip_landmarks = [lip_landmarks[0]]
    for lip_landmark in lip_landmarks[1:]:
        [rot_x, rot_y] = rotate_point((lip_landmark[0], 224-lip_landmark[1]), (lip_landmarks[0][0], 224-lip_landmarks[0][1]), angle_to_rotate_by)
        rotated_lip_landmarks.append([int(round(rot_x)), int(round(224-rot_y))])
    return np.array(rotated_lip_landmarks), angle_to_rotate_by


def rotate_point(point, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy]

