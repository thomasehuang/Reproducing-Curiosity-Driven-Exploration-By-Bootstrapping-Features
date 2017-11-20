#!/bin/sh

mkdir ~/.cloudshell/
touch ~/.cloudshell/no-apt-get-warning

sudo apt-get update 
sudo apt-get upgrade -y
sudo apt-get install -y python3-pip cmake libopenmpi-dev openmpi-bin zlib1g-dev libsm6 libxrender1 libfontconfig1 libxext6
pip3 install -r ../requirements.txt
sudo pip3 install --upgrade pip3
sudo pip3 install --upgrade tensorflow
sudo pip3 install opencv-python


git config --global user.email "audrow@umich.edu"
git config --global user.name "Audrow Nash"
