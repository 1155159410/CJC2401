# %% Import necessary modules
import warnings

# Suppress all Python warnings (global)
warnings.filterwarnings('ignore')

from sem2_classes import *


# %% Version 1: Sequential, no multithreading (~13 FPS; ~0.0339 s)
def v1():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    posture_system = PostureCorrectionSystem()
    frame_times = []

    while cv2.waitKey(1) != ord('q'):  # Up to 1000 loops per second
        ret, frame = cap.read()  # Read a frame
        if not ret:  # If frame read fails
            print("Failed to get frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        capture_time = time.time()

        result: dict = posture_system.process_image(rgb_frame)

        frame_processor = FrameProcessor(rgb_frame)
        if (keypoints := result.get('keypoints')) is not None:
            frame_processor.draw_skeletons(keypoints)
        rgb_frame = frame_processor.rgb_frame

        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Posture Correction System", bgr_frame)

        current_time = time.time()
        while frame_times and frame_times[0] < current_time - 1:
            frame_times.pop(0)
        frame_times.append(capture_time)
        fps = len(frame_times)

        print("Delay", current_time - capture_time)
        print("FPS", fps)
        print()

    cap.release()
    cv2.destroyAllWindows()

    posture_system.stop_thread()
    print("System stopped")
    del posture_system
    print("System deleted")


# %% Version 2: Multithreading (~27 FPS; ~0.0455 s)
def v2():
    in_queue: queue.Queue[dict] = queue.Queue(maxsize=1)  # Limit to at most 1 pending frame
    out_queue: queue.Queue[dict] = queue.Queue()

    camera = Camera()
    posture_system = PostureCorrectionSystem()

    camera.start(in_queue)
    posture_system.start_thread(in_queue, out_queue)

    frame_times = []

    while cv2.waitKey(1) != ord('q'):  # Up to 1000 loops per second
        item: dict = out_queue.get()

        frame_processor = FrameProcessor(item['rgb_frame'])
        if (keypoints := item.get('keypoints')) is not None:
            frame_processor.draw_skeletons(keypoints)
        rgb_frame = frame_processor.rgb_frame

        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Posture Correction System", bgr_frame)

        current_time = time.time()
        while frame_times and frame_times[0] < current_time - 1:
            frame_times.pop(0)
        frame_times.append(item['timestamp'])
        fps = len(frame_times)

        print("Delay", current_time - item['timestamp'])
        print("FPS", fps)
        print()

    cv2.destroyAllWindows()
    camera.stop()
    print("Camera stopped")
    posture_system.stop_thread()
    print("System stopped")
    del posture_system
    print("System deleted")


# %% Version 3: With rich overlays
def v3():
    in_queue: queue.Queue[dict] = queue.Queue(maxsize=1)  # Limit to at most 1 pending frame
    out_queue: queue.Queue[dict] = queue.Queue()

    camera = Camera()
    posture_system = PostureCorrectionSystem()

    camera.start(in_queue)
    posture_system.start_thread(in_queue, out_queue)

    # A list of frame timestamps for calculating FPS
    frame_times: list[float] = []

    # Main thread loop for displaying frames
    while cv2.waitKey(1) != ord('q'):  # Up to 1000 loops per second
        item: FrameInfo = out_queue.get()
        frame_processor = FrameProcessor(item['rgb_frame'])

        # Calculate FPS
        current_time = time.time()
        while frame_times and frame_times[0] < current_time - 1:
            frame_times.pop(0)
        frame_times.append(item['timestamp'])
        fps = len(frame_times)

        # Draw FPS
        frame_processor.put_fps(fps)

        if 'keypoints' in item:
            # Draw skeletons
            frame_processor.draw_skeletons(item['keypoints'])

            text = f"{item['predicted_posture']} ({item['posture_prob'].max():.4f})"
            frame_processor.put_text(text, color=(0, 255, 0), left=True)
            text = f"{item['predicted_feedback']} ({item['correctness_prob'][item['posture_prob'].argmax()]:.4f})"
            frame_processor.put_text(text, color=(0, 255, 0), left=False)

        # Display frame
        bgr_frame = frame_processor.bgr_frame
        cv2.imshow("Posture Correction System", bgr_frame)

        print("Delay", current_time - item['timestamp'])
        print("FPS", fps)
        print()

    # Clean up
    cv2.destroyAllWindows()
    camera.stop()
    print("Camera stopped")
    posture_system.stop_thread()
    print("System stopped")


# %% TODO: Majority voting

# %% Entry point
if __name__ == '__main__':
    # v1()
    # v2()
    v3()
