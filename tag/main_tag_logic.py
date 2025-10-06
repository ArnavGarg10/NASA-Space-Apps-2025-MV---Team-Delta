import time
import os
import board
import busio
import adafruit_adxl34x
import cv2
import subprocess
import geocoder
import csv
from datetime import datetime

# ------------------- Setup -------------------
# Accelerometer
i2c = busio.I2C(board.SCL, board.SDA)
accelerometer = adafruit_adxl34x.ADXL345(i2c)

# Webcam
cam = cv2.VideoCapture(0)

# Local folder for pictures
output_dir = "/home/pi/FoodPictures"
os.makedirs(output_dir, exist_ok=True)

# Initialize picture count from existing files
existing_files = [f for f in os.listdir(output_dir) if f.startswith("pic_") and f.endswith(".jpg")]
if existing_files:
    existing_numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
    picture_count = max(existing_numbers)
else:
    picture_count = 0

# Track which pictures have already been sent
last_sent_picture = picture_count

# ------------------- Remote (Buoy) info -------------------
remote_user = "nasa"
remote_host = "10.0.0.186"   # Your Linux PC IP
remote_dir = "/home/nasa/ReceivedPictures"

# ------------------- State tracking -------------------
last_capture_time = 0
open_streak = 0
close_streak = 0
taking_pictures = False
last_state = "CLOSED"

# ------------------- Helper functions -------------------
def get_state(x_value):
    """Return 'OPEN' or 'CLOSED' based on X-axis."""
    if x_value >= 1.75:
        return "OPEN"
    elif x_value <= 1.0:
        return "CLOSED"
    else:
        return None  # In between

def get_location():
    """Return approximate [latitude, longitude] using IP/Wi-Fi geolocation."""
    g = geocoder.ip('me')
    if g.ok:
        return g.latlng  # [lat, lng]
    else:
        return [None, None]

# ------------------- Main loop -------------------
while True:
    # Read accelerometer
    x, y, z = accelerometer.acceleration
    state = get_state(x)

    if state is None:
        state = last_state
    else:
        last_state = state

    # Track streaks
    if state == "OPEN":
        open_streak += 1
        close_streak = 0
    else:
        close_streak += 1
        open_streak = 0
        taking_pictures = False

    # Start taking pictures after 5 consecutive opens
    if open_streak >= 5:
        taking_pictures = True

    # ------------------- Take pictures -------------------
    if taking_pictures:
        interval = 1 if picture_count < 10 else 5  # seconds
        current_time = time.time()
        if current_time - last_capture_time >= interval:
            ret, frame = cam.read()
            if ret:
                picture_count += 1
                filename = os.path.join(output_dir, f"pic_{picture_count}.jpg")
                cv2.imwrite(filename, frame)
                print(f"ð¬ Saved {filename}")

                # Get location
                lat, lng = get_location()
                # Get timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Append location + timestamp to CSV
                csv_file = os.path.join(output_dir, "photo_locations.csv")
                write_header = not os.path.exists(csv_file)  # write header if file doesn't exist
                with open(csv_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(["filename", "timestamp", "latitude", "longitude"])
                    writer.writerow([f"pic_{picture_count}.jpg", timestamp, lat, lng])

                last_capture_time = current_time
            else:
                print("â ï¸ Failed to capture image from webcam!")

    # ------------------- Send only new pictures via SCP -------------------
    if close_streak >= 50:
        # Get new pictures
        new_files = [f for f in os.listdir(output_dir)
                     if f.startswith("pic_") and f.endswith(".jpg")
                     and int(f.split("_")[1].split(".")[0]) > last_sent_picture]
        if new_files:
            print(f"ð Closed 50+ times. Sending {len(new_files)} new pictures via SCP...")
            try:
                subprocess.run(
                    ["scp"] + new_files + ["photo_locations.csv"] + [f"{remote_user}@{remote_host}:{remote_dir}"],
                    cwd=output_dir,
                    check=True
                )
                print("â New pictures and CSV sent successfully!")

                # Delete sent pictures (keep CSV if desired)
                for f in new_files:
                    os.remove(os.path.join(output_dir, f))
                    print(f"ðï¸ Deleted sent picture: {f}")

                # Update last sent picture counter
                last_sent_picture = picture_count
            except subprocess.CalledProcessError as e:
                print("â ï¸ SCP failed:", e)
        else:
            print("â¹ï¸ No new pictures to send.")

        # Reset close streak after sending
        close_streak = 0

    # Debug info
    print(f"X: {x:.2f}, State: {state}, OpenStreak: {open_streak}, CloseStreak: {close_streak}, Pics: {picture_count}, LastSent: {last_sent_picture}")

    # Check every 0.25 seconds
    time.sleep(0.25)
