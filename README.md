# Tennis Video Analysis

<div>
    <img src="https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python"/>
</div>

# Introduction

The goal of this project is to build a python script to analyze a tennis video, to track the tennis ball, detect players and identify shots, using standard video processing techniques and a just little bit of ML (yolo). This work is based on a previous [existent work](https://github.com/DinosaurMauricio/tennis-ball-isv), actually being an enhancement of it.
The script keeps previous obtained results, adjusts some of them and performs new computations to a deeper analysis of the tennis video.

# Pre-existing features

- Tennis ball detection
- Trajectory drawing, for ball visual tracking
- Direction and in/out detection and printing

For more info, visit the [project GitHub](https://github.com/DinosaurMauricio/tennis-ball-isv).

# New Features

- Correction of tennis ball **direction** 
- Script to find the tennis field perimeter, using color (entire field, including alleys)
- Detect single field lines and **perimeter**, using line detection
- Projection of the tennis ball in a **2d field**, from a top view.
- Detection of bottom player's **shots** (backhand, forehand, serve) using YOLOv8 to detect his tennis racket
- Identification of the **type of shot** made by the bottom player, based on position and estimated direction (center, cross-court, down-the-line, inside-in, inside-out).

# Getting started

1. Initialize the workspace
    ```bash
    git clone https://github.com/luckeez/siv-project-tennis
    cd siv-project-tennis
    ```

2. Download [YOLOv8n.pt model](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt&ved=2ahUKEwih--ui14-FAxX4Q_EDHRcEDEwQFnoECBIQAQ&usg=AOvVaw0xT1jI0XjDZI-PC-WWmzci) and save it within the workspace just created.

3. Run the project.
    ```bash
    python3 src/main.py
    ```
    Arguments:
    - -v, --video: video path (default: "tennis_match.mp4"). 
    - -b, --buffer: max buffer size for trajectory draw (default: 64).
    - -y, --yolo: y/n to visualize or not yolo detection (default: "n").

    Example
    ```bash
    python3 src/main.py --video tennis_match_2.mp4 --buffer 32 --yolo y
    ```


