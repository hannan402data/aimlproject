# aimlproject
This project emphasizes about the face recognition application for food ordering management system. This project is developed by trained AI/ML Intern under GPRobo startup guidance.
Introduction
This project is a robot system that provides personalized food recommendations to users based on their face recognition and previous orders. The system uses a combination of computer vision, machine learning, and database management to offer a seamless and intuitive user experience.

Features
Face recognition using OpenCV and face_recognition libraries
Personalized food recommendations based on user's previous orders
User registration and login system
Database management using SQLite
Customizable UI using customtkinter library
Requirements
Python 3.8 or later
OpenCV 4.5 or later
face_recognition 1.3 or later
customtkinter 2.5 or later
SQLite 3.35 or later
numpy 1.20 or later
pandas 1.3 or later
scikit-learn 1.0 or later
Installation
Clone the repository using git clone https://github.com/your-username/robot-system.git
Install the required libraries using pip install -r requirements.txt
Create a new SQLite database using sqlite3 users.db
Run the application using python app.py
Usage
Run the application and click on the "Register" button to register a new user.
Enter your name, email, and date of birth, and click on the "Register" button.
The system will capture your face and save it to the database.
Once registered, you can login to the system using your face.
The system will display your previous orders and provide personalized food recommendations.
You can select from the recommended options and place an order.
Technical Details
The system uses the face_recognition library to detect and recognize faces.
The face encodings are stored in the SQLite database for future reference.
The system uses a machine learning model to generate personalized food recommendations based on the user's previous orders.
The UI is built using customtkinter library and is customizable.
Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.
