from predict import predict_on_images
import time
import sys

def countdown_timer(seconds):
    try:
        for remaining in range(seconds, 0, -1):
            mins, secs = divmod(remaining, 60)
            timer_str = f"Next run in: {mins:02}:{secs:02}"
            print(f"\r{timer_str}", end="")
            time.sleep(1)
        print("\r" + " " * len(timer_str), end="")  # Clear line after countdown
    except KeyboardInterrupt:
        print("\nCountdown interrupted.")

def main_loop():
    while True:
        print("\nRunning predictions...\n")
        predict_on_images(
            model_paths=["models/FishInv.pt", "models/MegaFauna.pt"],
            confs_threshold=[0.523, 0.546],
            images_input_folder_path="test_imgs/input/input/",
            images_output_folder_path="test_imgs/output/",
            delete_no_detections=True
        )
        countdown_timer(120)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nProcess stopped by user.")
