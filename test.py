import random
import time

# simulated heart rate data generator
def generate_heart_rate():
    return random.randint(60, 100)  # simulated heart rate between 60 and 100 BPM

# gets heart rate data every second
def fetch_heart_rate():
    while True:
        heart_rate = generate_heart_rate()
        print(f"Heart Rate: {heart_rate} BPM")
        time.sleep(1)

fetch_heart_rate()  # this would be where the api is called

# drowsiness based on heart rate
THRESHOLD_LOW = 65  # testing threshold for low heart rate indicating drowsiness, will figure out actual val later

def check_drowsiness(heart_rate):
    if heart_rate < THRESHOLD_LOW:
        return True
    else:
        return False

heart_rate = generate_heart_rate()  # Replace with actual fetched heart rate
if check_drowsiness(heart_rate):
    print("Drowsiness detected! Alerting user...")
    # Add code here to trigger the violent buzz
else:
    print("Heart rate within normal range.")


class AlertManager:
    def __init__(self):
        self.alert_active = False

    def start_alert(self):
        self.alert_active = True
        print("ALERT: Wake up! Drowsiness detected.")

    def stop_alert(self):
        self.alert_active = False
        print("Alert stopped.")

alert_manager = AlertManager()

# this will be called if drowsiness is detected
alert_manager.start_alert()