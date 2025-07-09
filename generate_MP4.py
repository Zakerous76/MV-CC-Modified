import cv2
import os
from tqdm import tqdm
datasets=['train','val','test']
for dataset in datasets:
    directory = f'./LEVIR-MCI-dataset/images/{dataset}'
    file_names = os.listdir(f"{directory}/A")
    video_dir = f'./LEVIR-MCI-dataset/images/{dataset}/video_data'
    os.makedirs(video_dir, exist_ok=True)


    for file_name in tqdm(file_names):
        name_without_extension, _ = os.path.splitext(file_name)
        image1 = cv2.imread(
            f'./LEVIR-MCI-dataset/images/{dataset}/A/{file_name}')
        image2 = cv2.imread(
            f'./LEVIR-MCI-dataset/images/{dataset}/B/{file_name}')
        if image1.shape[:2] != image2.shape[:2]:
            raise ValueError("Images must have the same size.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{video_dir}/{name_without_extension}.mp4', fourcc, 2.0,
                            (image1.shape[1], image1.shape[0]))
        for i in range(8):
            weight = i / 7.0  
            interpolated_frame = cv2.addWeighted(
                image2, weight, image1, 1 - weight, 0)
            out.write(interpolated_frame)
        out.release()


    # If I wanted to increase the size of the video
        # fps = 2.0
        # num_frames = 16  # More blended frames
        # hold_duration_seconds = 2  # Final frame hold time

        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # for i in range(num_frames):
        #     weight = i / (num_frames - 1)
        #     interpolated_frame = cv2.addWeighted(image2, weight, image1, 1 - weight, 0)
        #     out.write(interpolated_frame)

        # # Hold final frame
        # for _ in range(int(fps * hold_duration_seconds)):
        #     out.write(interpolated_frame)

        # out.release()
