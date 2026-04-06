import psutil

info = psutil.cpu_count(logical=False)  # nucleos fisicos
logical = psutil.cpu_count(logical=True)  # nucleos logicos
print(f"Fisicos: {info}, Logicos: {logical}")