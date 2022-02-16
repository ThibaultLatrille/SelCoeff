#!/usr/bin/env bash
git clone https://github.com/ThibaultLatrille/AdaptaPop
cd AdaptaPop
sudo apt install -qq -y python3-dev python3-pip
pip3 install --user scipy numpy matplotlib pandas
sudo apt install -qq -y snakemake
