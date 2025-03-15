# RadCare Assistant

## Description
RadCare Assistant is a tool that helps analyze chest X-ray images using machine learning. It can detect various conditions from X-ray images and provide quick analysis to support healthcare professionals.

## Main Features
- Upload and analyze chest X-ray images
- Automated detection of common chest conditions
- User-friendly interface for medical professionals
- Secure storage of analysis results

## Installation
1. Make sure you have Docker installed on your computer
2. Clone this repository
3. Build the Docker image:
   ```
   docker build -t radcare_assistant .
   ```
4. Run the container:
   ```
   docker run -p 8000:8000 radcare_assistant
   ```

## Basic Usage
1. Open the application in your web browser at http://localhost:8000
2. Upload a chest X-ray image
3. Wait for the analysis to complete
4. View the results and recommendations

## Requirements
- Docker
- Internet connection for initial setup

