#!/bin/bash
# this allows camera to work on Raspberry PI
# This program is automatically executed the the application at load time
# this application automatically is loaded at boot time so this command is automatically executed too.
sudo modprobe bcm2835-v4l2
