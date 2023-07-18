import cv2 as cv

vod = cv.VideoCapture(0)

ret, frame = vod.read()

gpu_frame = cv.cuda_GpuMat()

while ret:
    gpu_frame.upload(frame)

    frame = cv.cuda.resize(gpu_frame, (352, 288))
    frame.download()

    ret, frame = vod.read()