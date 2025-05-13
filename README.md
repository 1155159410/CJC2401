# CJC2401: Real-Time Posture Correction for Yoga Practice

**Author:** Wang Chi HUI  
**Affiliation:** The Chinese University of Hong Kong  
**Supervisor:** Dr. CHAU Chuck-jee

**[📄 View the full final report (PDF)](https://raw.githubusercontent.com/1155159410/CJC2401/main/miscellaneous/cjc2401_1155159410_final_report.pdf)**

## Abstract

This project aims to develop a real-time yoga posture correction system, addressing the lack of existing applications that provide instant feedback for solo practitioners, thus reducing injury risk. The system employs a two-stage approach: first, it uses BlazePose, a pre-trained pose estimation model, to extract body keypoints from live video streams; second, a custom-built Shared MLP deep learning model classifies four yoga postures (Down Dog, Plank, Side Plank, Warrior II) and evaluates their correctness.  
To support training, we compiled a dataset of over 3,800 labeled yoga images from online sources, including both correct and incorrect posture examples. The resulting model achieved approximately 93% accuracy on test data. Integrated into a complete system with a simple graphical interface, it provides smooth real-time feedback at around 30 FPS. Practical evaluations further demonstrated the system's near-perfect stability, enhanced using majority voting.  
Additionally, we explored a cutting-edge Vision-Language Model (VLM)-based approach, concluding that current VLM solutions, while promising in eliminating extensive dataset preparation, remain impractical due to significant computational overhead and limited accuracy in fine-grained classification tasks.

## Environment

- **Python version**: 3.10
- **Architecture**: ARM64
- **Platform**: macOS 15 Sequoia
