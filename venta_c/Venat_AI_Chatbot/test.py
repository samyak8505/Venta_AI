import subprocess

with open("requirements.txt") as f:
    packages = [line.strip() for line in f if line.strip()]

for pkg in packages:
    result = subprocess.run(
        ["pip", "show", pkg],
        capture_output=True,
        text=True
    )
    name, version = None, None
    for line in result.stdout.splitlines():
        if line.startswith("Name:"):
            name = line.split(":")[1].strip()
        elif line.startswith("Version:"):
            version = line.split(":")[1].strip()
    if name and version:
        print(f"{name}=={version}")
