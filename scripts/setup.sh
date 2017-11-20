#!/bin/sh

mkdir ~/.cloudshell/
touch ~/.cloudshell/no-apt-get-warning

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y python3-pip cmake libopenmpi-dev openmpi-bin
pip3 install -r ../requirements.txt
sudo pip3 install --upgrade pip3
sudo pip3 install --upgrade tensorflow
sudo pip3 install opencv-python

git clone https://github.com/openai/baselines.git
sudo pip3 install -e baselines
