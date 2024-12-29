# AID3 - 3D Print Failure Detection

## Overview
AID3 (Artificial Intelligence for 3D Printing Errors) is a cutting-edge project aimed at detecting failures in 3D prints using advanced computer vision techniques. Leveraging the YOLOv8 model, the system identifies anomalies in real-time from video feeds of the printing process, enabling early intervention and reducing material waste.

## Features
- **Live Monitoring**: Real-time failure detection using a video feed of the 3D printing process.
- **Custom Dataset**: Utilizes a labeled dataset of "GOOD" and "BAD" 3D prints to train the AI model for high accuracy.
- **Hyperparameter Optimization**: Fine-tunes YOLOv8 for enhanced detection performance.
- **Automated Failure Reporting**: Logs and reports failures as soon as they are detected during the printing process.

## Project Structure
The project is organized as follows:
- **Dataset**: Images from 3D printing timelapses, categorized into folders by print job and labeled as "GOOD" or "BAD."
- **Model Training**: A YOLOv8 model trained on the labeled dataset, optimized for accuracy and speed.
- **Live Testing**: Script to evaluate the model in real-time, detecting failures during the printing process.

## How It Works
1. **Data Preparation**: Images of 3D prints are categorized into folders based on whether they depict successful or failed prints. 
2. **Model Training**: The YOLOv8 model is trained using this dataset, incorporating bounding box annotations for precise detection.
3. **Real-Time Detection**: The trained model processes a live video feed to identify failures during the printing process.
4. **Failure Logging**: The system logs the timestamp and type of failure detected for further analysis.

## Applications
- **Reduced Waste**: Early failure detection minimizes wasted materials and time.
- **Increased Efficiency**: Automates the monitoring process, enabling hands-off operation.
- **Enhanced Quality Control**: Provides insights into common failure modes, improving print success rates over time.

## Future Improvements
- Integration with printer control systems for automatic pausing or adjustments upon failure detection.
- Expansion of the dataset to include more diverse failure types and printer models.
- Deployment as a standalone application for broader accessibility.

## Contact
For inquiries or contributions, please contact **Kase Johnson** at [kasejohnson01@gmail.com].
