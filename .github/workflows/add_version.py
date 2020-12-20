# This script updates package version while deploying
import sys

new_version = str(sys.argv[1])

with open("setup.py") as f:
    lines = f.readlines()
lines[4] = "version = '" + new_version[1:] + "'\n"

with open("setup.py", "w") as f:
    f.writelines(lines)
