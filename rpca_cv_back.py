import cv2;
help cv2.VideoCapture
movie = cv2.VideoCapture("RobustPCA_video_demo.avi")
frames= movie.get(cv2.CAP_PROP_FRAME_COUNT)
fps   = movie.get(cv2.CAP_PROP_FPS)
width = movie.get(cv2.CAP_PROP_FRAME_WIDTH)
height= movie.set(cv2.CAP_PROP_FRAME_HEIGHT)
print width, height, frames, fps

