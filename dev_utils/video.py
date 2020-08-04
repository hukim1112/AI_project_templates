import cv2

def video_info(input_video):
    vcap = cv2.VideoCapture(input_video)
    if vcap.isOpened():
        width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        # print(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT) # 3, 4

        fps = vcap.get(cv2.CAP_PROP_FPS)
        # print(cv2.CAP_PROP_FPS) # 5

        num_frames = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
        # print(cv2.CAP_PROP_FRAME_COUNT) # 7
        return int(height), int(width), fps, num_frames
    else:
        raise ValueError("vcap is not opened")

def video_process(input_video, output_video, start_frame=None, end_frame=None):
    height, width, fps, num_frames = video_info(input_video)
    video_shape = (height, width)
    assert end_frame - start_frame > 0, "end_frame value must be larger than start_frame"
    vcap = cv2.VideoCapture(input_video)
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    if start_frame is not None:
        vcap.set(1, start_frame)
        if end_frame is not None:
            print("frames are predicted from starting frame:{} to end frame:{}, batch_size:{}".format(start_frame, end_frame, frame_batch))
            record_length = end_frame - start_frame
        else:
            print("frames are predicted from starting frame:{} to the last frame:{} of video, batch_size:{}".format(start_frame, num_frames, frame_batch))
            record_length = num_frames - start_frame
    else:
        if end_frame is not None:
            print("frames are predicted from first_frame to end_frame:{} batch_size:{}".format(end_frame, frame_batch))
            record_length = end_frame
        else:
            print("frames are predicted all frames of video. The number of frames are {}".format(num_frames))
            record_length = num_frames

    output_format = os.path.splitext(output_video)[1]
    if output_format == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    elif output_format == '.avi':
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
    else:
        raise ValueError("video you want to write is not mp4 or avi.")

    vwriter = cv2.VideoWriter(output_annotated_video, fourcc, fps, video_shape)
    while(vcap.isOpened()):
        ret, frame = vcap.read()
        if ret == True and i < record_length:
            # new_frame = process(frame)
            # vwriter.write(new_frame)
            pass
        else:
            pass

    print("time to process this video : {}, second per frame : {}".format(
        end - start, (end - start) / record_length))

    vcap.release()
    vwriter.release()