for dir_name in */;
do
name=${dir_name%?};
echo "${name}";
# Delete old landmarks
rm ${name}/*.mat;
# Copy new landmarks
echo "Copying landmarks from /tmp";
echo "cp /tmp/landmarks/genKp/${name}_generated_lip_landmarks.mat ${name}/${name}_hindi_abhishek_generated_lip_landmarks.mat;"
cp /tmp/landmarks/genKp/${name}_generated_lip_landmarks.mat ${name}/${name}_hindi_abhishek_generated_lip_landmarks.mat;
:'
# Copy audio files
echo "Copying wav file";
cp /tmp/landmarks/genKp/new_hindi/test_wav/${name}.wav ${name}/${name}_hindi_abhishek.wav;
# Copy original video file
echo "Copying original mp4 file";
cp ../OLD/${name}/${name}.mp4 ${name}/;
'
echo "Making hindi ANDREW_NG video";
:'
# Generate new video, also generate making_video (4 frames in 2x2 grid)
python /users/abhishek/lipreading/visual_dub/movie-translation-experiments/andrew_ng/morph_video_with_new_lip_landmarks.py "${name}/${name}.mp4" -a "${name}/${name}_hindi_abhishek.wav" -l "${name}/${name}_hindi_abhishek_generated_lip_landmarks.mat" -o "${name}/${name}_hindi_abhishek.mp4" -m -v -y;
# Generate new video with landmarks on same original video "CV_01_C4W1L01_000003_to_000045.mp4", and generate making_video, and replace unvoiced segments with cluster center of closed mouth
python /users/abhishek/lipreading/visual_dub/movie-translation-experiments/andrew_ng/morph_video_with_new_lip_landmarks.py "CV_01_C4W1L01_000003_to_000045/CV_01_C4W1L01_000003_to_000045.mp4" -a "${name}/${name}_hindi_abhishek.wav" -l "${name}/${name}_hindi_abhishek_generated_lip_landmarks.mat" -o "${name}/${name}_hindi_abhishek.mp4" -vadthresh 0.9 -m -r -y -v;
# Generate new video, also generate making_video, and replace unvoiced segments with cluster center of closed mouth
python /users/abhishek/lipreading/visual_dub/movie-translation-experiments/andrew_ng/morph_video_with_new_lip_landmarks.py "${name}/${name}.mp4" -a "${name}/${name}_hindi_abhishek.wav" -l "${name}/${name}_hindi_abhishek_generated_lip_landmarks.mat" -o "${name}/${name}_hindi_abhishek.mp4" -vadthresh 0.9 -r -m -y -v;
'
# Generate new video, also generate making_video
python /users/abhishek/lipreading/visual_dub/movie-translation-experiments/andrew_ng/morph_video_with_new_lip_landmarks.py "${name}/${name}.mp4" -a "${name}/${name}_hindi_abhishek.wav" -l "${name}/${name}_hindi_abhishek_generated_lip_landmarks.mat" -o "${name}/${name}_hindi_abhishek.mp4" -m -y -v;
done

# Make videos of lip landmark plots (graphs)
for dir_name in */;
do
name=${dir_name%?}
echo "${name}";
echo "Making video of landmarks";
python /users/abhishek/lipreading/visual_dub/movie-translation-experiments/lip_landmarks_experiments/make_video_of_lip_landmarks.py "${name}/${name}_hindi_abhishek_generated_lip_landmarks.mat" -a "${name}/${name}_hindi_abhishek.wav" -o "${name}/${name}_hindi_abhishek_generated_lip_landmarks.mp4" -y;
done

