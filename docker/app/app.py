import datetime
import time

print("Hello from the Docker container!")
print(f"Current date and time: {datetime.datetime.now()}")

## Keep the container running to demonstrate it's working
print("Container is running...")
while True:
    print("Container still running... Press Ctrl+C to stop.")
    time.sleep(10)