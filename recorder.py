import sys
import os

print("arecord -D hw:1,0 -c 1 -r 48000 -d " + sys.argv[2] + " -f S16_LE " + sys.argv[1] + ".wav")
os.system("arecord -D hw:1,0 -c 1 -r 48000 -d " + sys.argv[2] + " -f S16_LE " + sys.argv[1] + ".wav")
os.system("aplay " + sys.argv[1] + ".wav")

