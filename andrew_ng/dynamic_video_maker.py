import cv2
import numpy as np
import os
import scipy.io as sio
import sys
import tqdm

from sklearn.neighbors import NearestNeighbors


def vid2img(filename):
    cap = cv2.VideoCapture(filename)
    success, frame = cap.read()
    allImg = []
    while(success):
        allImg.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        success, frame = cap.read()
    return allImg


def allImg2vid(allImg, output_vid='output_body.mp4', frameRate=30):
    vidCodec = cv2.VideoWriter_fourcc(*'XVID')
    height, width, channel = allImg[0].shape
    vidFile = cv2.VideoWriter(output_vid, vidCodec, frameRate, (width, height))
    for i in tqdm.tqdm(range(0, len(allImg))):
        frame = cv2.cvtColor(allImg[i], cv2.COLOR_RGB2BGR)
        vidFile.write(frame)
    vidFile.release()
    print("Successfully converted images in allImg to " + output_vid)


def get_dynamic_video(target_video_frames, target_video_landmarks, source_lip_landmarks, pred_lm_fps, audio_filename, save_output_video=False, output_filename='out_with_jaw_body.mp4'):
#     input_pred_lm_filename is the preddicted 20D landmarks .mat  file format with y_pred as the key for landmarks
#     target_lm_filename is the target video 68D landmarks in .mat file format with ypred as the key for the landmarks
#     default pred landmark fps = 25
#     input_video_filename is the video corresponding to input_pred_lm_filename
#     target_video_filename is the target video filename
    
    # Read target_video_lm
    # print('reading file:'+target_lm_filename)
    # full_lm_dst = sio.loadmat(target_lm_filename)['ypred']
    full_lm_dst = target_video_landmarks
    lm_dst = full_lm_dst[:, -20:]
    lm_center_dst = lm_dst[:, [14, 16]]
    lower_lm_dst = lm_center_dst[:, 1, 0]
    lower_lm_dst = (lower_lm_dst-lower_lm_dst.min())/(lower_lm_dst.max()-lower_lm_dst.min())
    
    # Read source lm
    # print('reading file:'+input_pred_lm_filename)
    # lm_src = sio.loadmat(input_pred_lm_filename)['y_pred']
    lm_src = source_lip_landmarks
    lm_center_src = lm_src[:, [14, 16]]
    lower_lm_src = lm_center_src[:, 1, 0]
    lower_lm_src = (lower_lm_src-lower_lm_src.min())/(lower_lm_src.max()-lower_lm_src.min())
    
    # Define cluster centroids
    k_centers = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype('float')
    
    # Find nearest neighbour frame index in target video of each cluster centroid
    neigh = NearestNeighbors(1)
    neigh.fit(np.expand_dims(lower_lm_dst, 1))
    dst_centroids = neigh.kneighbors(np.expand_dims(k_centers, 1), 1, return_distance=False)
    
    # Map source landmarks to nearest frame of cluster centroid
    neigh = NearestNeighbors(1)
    neigh.fit(np.expand_dims(k_centers, 1))
    src_assignment = neigh.kneighbors(np.expand_dims(lower_lm_src, 1), 1, return_distance=False)
    
    dst_frame_ids_for_source = dst_centroids[src_assignment].squeeze()
    temp = []
    for i in range(0, 5):
        temp.append(dst_frame_ids_for_source[0])
    dst_frame_ids_for_source = np.array(temp + dst_frame_ids_for_source.tolist())
    
    # print('reading file:' + target_video_filename)
    # allImg_im1 = vid2img(target_video_filename)
    allImg_im1 = target_video_frames
    allImg_dst2src = []
    dynamic_lm_dst = []
    for i in dst_frame_ids_for_source:
        allImg_dst2src.append(allImg_im1[i])
        dynamic_lm_dst.append(full_lm_dst[i])
    
    dynamic_lm_dst = np.array(dynamic_lm_dst)
    # sio.savemat('Dynamic_LM.mat', {'lm': dynamic_lm_dst})
    if save_output_video:
        allImg2vid(allImg_dst2src, '/tmp/output_with_jaw.mp4', frameRate=pred_lm_fps)
        shell_command = 'ffmpeg -i /tmp/output_with_jaw.mp4 -i ' + audio_filename + ' -c copy -map 0:v:0 -map 1:a:0 -shortest ' + output_filename
        os.system(shell_command)

    # Return dyn_sync frames and corresponding landmarks
    return allImg_dst2src, dynamic_lm_dst


# get_dynamic_video(pred_lm_fps=25, input_pred_lm_filename='CV_01_C4W1L01_000003_to_000045_hindi_abhishek_generated_lip_landmarks.mat', target_lm_filename='ANDREW_NG_CV_01_C4W1L01_000003_to_000045_landmarks_in_frames.mat', input_video_filename='input_videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045_hindi_abhishek_making.mp4', target_video_filename='input_videos/CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045.mp4')

