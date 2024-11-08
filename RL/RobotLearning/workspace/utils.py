import cv2

def save_video(frames, save_path, fps=30):
    height, width, layers = frames[0].shape
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(* "mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, size)
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()