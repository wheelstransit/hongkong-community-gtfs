import requests
import threading
import time

URL = "https://data.etagmb.gov.hk/stop/20001397"
NUM_THREADS = 50
SUCCESS_STATUS_CODE = 200
RATE_LIMIT_STATUS_CODE = 403

# --- Shared State for Threads ---
# A thread-safe counter for successful requests
success_count = 0
request_lock = threading.Lock()

# A thread-safe event to signal all threads to stop
rate_limit_hit_event = threading.Event()

def make_requests():
    """
    This function will be executed by each thread.
    It continuously sends requests until the rate_limit_hit_event is set.
    """
    global success_count
    
    # Keep making requests as long as the stop signal has not been received
    while not rate_limit_hit_event.is_set():
        try:
            # Add a timeout to prevent threads from hanging indefinitely
            response = requests.get(URL, timeout=10)

            # Check if another thread has already hit the rate limit while we were waiting
            if rate_limit_hit_event.is_set():
                break

            if response.status_code == SUCCESS_STATUS_CODE:
                # Use a lock to safely increment the shared counter
                with request_lock:
                    success_count += 1
                    # To avoid spamming the console, only print every 50 requests
                    if success_count % 50 == 0:
                        print(f"Success count: {success_count}...")
            
            elif response.status_code == RATE_LIMIT_STATUS_CODE:
                print("\n" + "="*40)
                print(f"RATE LIMIT HIT! Status Code: {response.status_code}")
                print("="*40)
                # Set the event to signal all other threads to stop
                rate_limit_hit_event.set()
                # Break the loop for this thread
                break
                
            else:
                # Handle other unexpected status codes
                print(f"\n[!] Received unexpected status code: {response.status_code}")
                # You might want to stop on other errors too
                rate_limit_hit_event.set()
                break

        except requests.exceptions.RequestException as e:
            print(f"\n[!] An error occurred: {e}")
            # In case of network errors, we might want to stop the test
            rate_limit_hit_event.set()
            break


if __name__ == "__main__":
    print(f"[*] Starting rate limit test against: {URL}")
    print(f"[*] Using {NUM_THREADS} concurrent threads.")
    print("[*] The test will stop when a 403 Forbidden status is received.")
    print("-" * 60)

    threads = []
    start_time = time.time()

    # Create and start the threads
    for _ in range(NUM_THREADS):
        thread = threading.Thread(target=make_requests)
        thread.daemon = True  # Allows main thread to exit even if threads are blocking
        thread.start()
        threads.append(thread)

    # You can either wait for the event or join the threads.
    # Waiting for the event is cleaner if you want to report results immediately.
    rate_limit_hit_event.wait()
    
    # Or you could wait for all threads to finish their current loop
    # for thread in threads:
    #     thread.join()

    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "-" * 60)
    print("Test Finished.")
    print(f"Total successful requests before rate limit: {success_count}")
    print(f"Test duration: {duration:.2f} seconds")
    print("-" * 60)