# security_AI
AI-Powered Multimodal Biometric Security System


A Python-based security and access-control project exploring the application of AI in biometric identity verification. The system combines face and voice recognition pipelines with multimodal decision logic to evaluate identity matches and determine whether access should be granted or denied.

The project features a modular architecture for biometric data preprocessing, feature embedding management, similarity-based matching, and authentication, alongside a Streamlit dashboard for monitoring results and reviewing security events.

Key Features
Face Recognition: Processes facial data and uses face embeddings for identity matching.
Voice Recognition: Uses voice samples and voice embeddings to support speaker identification.
Multimodal Authentication: Combines facial and voice matching results to support access-control decisions.
Similarity-Based Matching: Supports biometric comparison using configurable thresholds and distance metrics.
Security Monitoring Dashboard: Visualizes authentication outcomes, confidence scores, alerts, and event history through Streamlit.
Simulation and Testing: Demonstrates authorized and unauthorized access scenarios.
Modular Python Architecture: Separates recognition pipelines, authentication logic, data management, preprocessing, and logging.
Technologies
Python · Streamlit · OpenCV · NumPy · Pandas · AI/ML-based biometric processing

Project Goals
This project explores the practical application of AI techniques to identity verification, multimodal biometric processing, and security automation. It also emphasizes clean code organization, reusable modules, configurable parameters, and the integration of recognition pipelines into a complete application.

Note: The current dashboard uses simulated biometric inputs for demonstration. The actual recognition capabilities depend on the implementations of the underlying face and voice modules.
