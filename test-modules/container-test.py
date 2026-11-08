from time import sleep
from urllib.request import urlopen
from urllib.error import URLError
import subprocess
from pathlib import Path


url = "http://127.0.0.1:88/health"

# checking for container startup initiation
for attempt in range(1, 11):
    try:
        with urlopen(url, timeout=10) as response:
            result = response.read()
            Path("health-response.json").write_bytes(result)
            print("Container is ready")
            break

    except (URLError, TimeoutError,OSError) as error:
        print(f"waiting for container: attempt {attempt}/10")
        print(f"Reason: {error}")
        sleep(3)

else:
    print("Container did not become ready")
    subprocess.run(["docker", "logs", "loan-service"], check=False)
    raise SystemExit(1)