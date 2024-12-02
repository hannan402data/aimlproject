import customtkinter as ctk
import tkinter as tk
from CTkMessagebox import CTkMessagebox
from tkinter import messagebox, simpledialog
import cv2
import face_recognition
import numpy as np
import sqlite3
import threading
import queue
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from dateutil import parser
import nltk
nltk.download('punkt_tab')

class FoodRecommender:
    def __init__(self, cursor):
        self.c = cursor
        self.food_data = pd.read_csv("indian_food.csv")  
        self.food_df = pd.DataFrame(self.food_data, columns=["Name", "Ingredients", "Veg_Non", "Flavour", "Course"])

        mlb = MultiLabelBinarizer()

        self.food_df['Ingredients'] = self.food_df['Ingredients'].str.split(',').apply(lambda x: [i.strip() for i in x])
        one_hot = mlb.fit_transform(self.food_df['Ingredients'])

        one_hot_df = pd.DataFrame(one_hot, columns=mlb.classes_)
        self.final_df = pd.concat([self.food_df['Name'], one_hot_df], axis=1)
        self.cosine_sim = cosine_similarity(self.final_df.drop(columns=['Name']))

    
    # def generate_recommendations(self, user_id):  
   # Load the user's previous orders  
        # self.c.execute("SELECT orders FROM users WHERE id = ?", (user_id,))  
        # previous_orders_row = self.c.fetchone()

        # if previous_orders_row is None:  
        #     return []  # Return an empty list if there are no previous orders  
        
        # previous_orders = previous_orders_row[0]
        
        # # Split the previous orders into a list  
        # previous_orders_list = previous_orders.split(', ')  
        
        # # Filter the food data to only include dishes that are not in the previous orders  
        # filtered_food_data = self.food_df[~self.food_df["Name"].isin(previous_orders_list)]  
        
        # # Convert the list of ingredients into a string  
        # filtered_food_data["Ingredients"] = filtered_food_data["Ingredients"].apply(lambda x: ', '.join(x))  
        
        # # Create a TF-IDF vectorizer  
        # vectorizer = TfidfVectorizer()  
        
        # # Fit the vectorizer to the ingredients and transform the data  
        # ingredients_vectors = vectorizer.fit_transform(filtered_food_data["Ingredients"])  
        
        # # Transform the previous orders into vectors  
        # previous_orders_vectors = vectorizer.transform(previous_orders_list)  
        
        # # Calculate the cosine similarity between the ingredients and the previous orders  
        # similarity_matrix = cosine_similarity(ingredients_vectors, previous_orders_vectors)  
        
        # # Get the top 5 recommendations based on the similarity matrix  
        # recommendations = filtered_food_data.iloc[similarity_matrix.argsort().flatten()[-5:]]  
        
        # return recommendations["Name"].tolist()

    def generate_recommendations(self, name):
        indices = pd.Series(self.final_df.index, index=self.final_df['Name']).drop_duplicates()
        indices
        idx = indices[name]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        dish_indices = [i[0] for i in sim_scores]
        return self.final_df['Name'].iloc[dish_indices]

    def text_cleaning(text):
        describe_text = word_tokenize(text)
        return describe_text
        
    

class RecogniseFaceOrderFood(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.conn = sqlite3.connect('users.db', check_same_thread=False)  
        self.c = self.conn.cursor()
        self.recommender = FoodRecommender(self.c)

        self.initialize_database()

        self.title("Robot System for Personalized Food Recommendations")
        self.geometry("500x400")

        self.face_rec_sys = ctk.CTkLabel(self, text="GPRobo", font=("Arial", 24, "bold"))
        self.face_rec_sys.pack(pady=20)

        self.msg = ctk.CTkLabel(self, text="Please look into the camera for recognition", font=("Arial", 16))
        self.msg.pack(pady=10)

        self.loading_label = ctk.CTkLabel(self, text="Detecting...", font=("Arial", 12))
        self.loading_label.pack(pady=10)

        self.user_queue = queue.Queue()
        self.current_user_processing = False

        self.queue = queue.Queue()
        self.camera_thread = threading.Thread(target=self.show_camera_feed)
        self.camera_thread.start()

        self.reload_button = ctk.CTkButton(self, text="Reload", command=self.reload_app)
        self.reload_button.pack(pady=10)

        self.after_id = None
        self.process_queue()

        self.mainloop()

    def initialize_database(self):
        """Initialize the database schema"""
        self.c.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT,
                            email TEXT,
                            face_encoding BLOB,
                            consent INTEGER DEFAULT 0,
                            orders TEXT,
                            dob TEXT)''')
        self.c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            feedback_text TEXT,
                            FOREIGN KEY (user_id) REFERENCES users (id))''')
        self.c.execute('''CREATE TABLE IF NOT EXISTS food (
                            Food_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                            Name TEXT,
                            Veg_Non TEXT,
                            Ingredients TEXT)''')
        self.conn.commit()

    def encode_face(self, image_path):
        """Encode the face from the image path"""
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        return encoding[0] if encoding else None
    
    def show_camera_feed(self):
        """Continuously capture face from camera feed until face is detected"""
        cv2.namedWindow("Camera - Face Detection", cv2.WINDOW_NORMAL) 
        cv2.moveWindow("Camera - Face Detection", 100, 100)  

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.queue.put("Camera not found!")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.queue.put("Failed to capture image from camera.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                rgb_frame = frame[:, :, ::-1]
                face_locations = face_recognition.face_locations(rgb_frame, model='hog')

                if face_locations:
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Save the face and queue the encoding for further processing
                    cv2.imwrite("user_detected.jpg", frame)
                    encoding = self.encode_face("user_detected.jpg")
                    if encoding is not None:
                        # Queue the face encoding if there's a current user being processed
                        if not self.current_user_processing:
                            self.queue.put(encoding)
                            self.current_user_processing = True # Lock processing for current user
                            break  
                        else:
                            self.user_queue.put(encoding)  # Put new user in queue
                        
                    cv2.imshow("Camera - Face Detection", frame)
                        
                else:
                    cv2.putText(frame, "Please look into the camera properly", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow("Camera - Face Detection", frame)

                if cv2.waitKey(3) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def process_queue(self):
        """Process the queue for messages or encoding"""
        try:
            item = self.queue.get_nowait()
            if isinstance(item, str):
                messagebox.showerror("Error", item)
            else:
                self.process_face(item)
        except queue.Empty:
            self.after_id = self.after(100, self.process_queue)

    def process_face(self, encoding):
        """Process the detected face (login or register)"""
        def callback():
            self.loading_label.pack_forget()
            self.c.execute("SELECT id, name, face_encoding, consent FROM users")
            users = self.c.fetchall()

            user_found = False

            for user in users:
                user_id, user_name, user_encoding, consent = user
                known_encoding = np.frombuffer(user_encoding, dtype=np.float64)

                matches = face_recognition.compare_faces([known_encoding], encoding)
                if matches[0]:
                    user_found = True
                    self.initial_greeting(user_name, consent, user_id)
                    return

            if not user_found:
                self.enroll_new_customer(encoding)
    
        self.after(0, callback)
        
    def initial_greeting(self, user_name, consent, user_id):
        """Greet returning users or ask for consent if they haven't given it yet"""
        if consent == 1:
            CTkMessagebox(self, title="Login", message=f"Hello {user_name}! Welcome back to GPRobo.")
            self.show_home_screen(user_name, user_id)
        else:
            response = CTkMessagebox(self, title="Notice", message=f"{user_name}, please give consent for personalized recommendations.")
            if response:
                self.get_consent(user_name, user_id)

    def enroll_new_customer(self, encoding):
        """Greet new customers and invite them to join the personalized recommendation program"""
        CTkMessagebox(title="Welcome", message="Hello and welcome to GPRobo! We're excited to have you here.")
        
        register_prompt = CTkMessagebox(title="Join Program", message="Would you like to join our personalized food recommendation program?")
        if register_prompt:
            self.register_user(encoding)
        else:
            CTkMessagebox(title="Notice", message="Feel free to enjoy your meal. If you change your mind, let us know!")

    def get_consent(self, user_name, user_id):
        """Ask the user for consent to store and use their data for recommendations"""
        consent_window = ctk.CTkToplevel(self)
        consent_window.title("Consent Form")
        consent_window.geometry("400x200")

        consent_text = ctk.CTkLabel(consent_window, text=f"{user_name}, we value your privacy. To provide personalized recommendations, we need your consent to store and use your data. Do you agree?", font=("Arial", 12), text_color="#fff")
        consent_text.pack(pady=10)

        def give_consent():
            self.c.execute("UPDATE users SET consent = 1 WHERE id = ?", (user_id,))
            self.conn.commit()
            CTkMessagebox(title="Thank You", message="Thank you for providing your consent!")
            consent_window.destroy()

        consent_button = ctk.CTkButton(consent_window, text="Give Consent", command=give_consent, corner_radius=10)
        consent_button.pack(pady=20)

        def decline_consent():
            CTkMessagebox(title="Notice", message="Without consent, you won't be able to access personalized recommendations.")
            consent_window.destroy()

        consent_button = ctk.CTkButton(consent_window, text="Give Consent", command=give_consent, corner_radius=10)
        consent_button.pack(pady=10)
        decline_button = ctk.CTkButton(consent_window, text="Decline", command=decline_consent, corner_radius=10)
        decline_button.pack(pady=10)

    def register_user(self, encoding):
        """Register a new user and prompt for additional details"""
        registration_window = ctk.CTkToplevel(self)
        registration_window.title("Register")
        registration_window.geometry("400x300")

        name_label = ctk.CTkLabel(registration_window, text="Name:")
        name_label.pack(pady=5)
        name_entry = ctk.CTkEntry(registration_window)
        name_entry.pack(pady=5)

        email_label = ctk.CTkLabel(registration_window, text="Email:")
        email_label.pack(pady=5)
        email_entry = ctk.CTkEntry(registration_window)
        email_entry.pack(pady=5)

        dob_label = ctk.CTkLabel(registration_window, text="Date of Birth (YYYY-MM-DD):")
        dob_label.pack(pady=5)
        dob_entry = ctk.CTkEntry(registration_window)
        dob_entry.pack(pady=5)

        def complete_registration():
            name = name_entry.get()
            email = email_entry.get()
            dob = dob_entry.get()

            try:
                parsed_dob = parser.parse(dob)
                formatted_dob = parsed_dob.strftime('%Y-%m-%d')
            except ValueError:
                formatted_dob = None

            self.c.execute("INSERT INTO users (name, email, face_encoding, orders, dob) VALUES (?, ?, ?, ?, ?)",
                           (name, email, encoding.tobytes(), '', formatted_dob))
            self.conn.commit()
            user_id = self.c.lastrowid
            CTkMessagebox(title="Registration Successful", message="You have been successfully registered.")
            registration_window.destroy()
            self.show_home_screen(name, user_id)

        register_button = ctk.CTkButton(registration_window, text="Register", command=complete_registration)
        register_button.pack(pady=10)

    def show_home_screen(self, user_name, user_id):
        """Display the home screen with food order options"""
        home_screen = ctk.CTkToplevel(self)
        home_screen.title("Home Screen")
        home_screen.geometry("400x300")

        welcome_label = ctk.CTkLabel(home_screen, text=f"Welcome, {user_name}!", font=("Arial", 16))
        welcome_label.pack(pady=20)

        self.c.execute("SELECT orders FROM users WHERE id = ?", (user_id,))
        previous_orders = self.c.fetchone()[0]

        def feedback_provided():
            feedback = simpledialog.askstring("Feedback", "Please provide your feedback:")
            if feedback:
                self.c.execute("INSERT INTO feedback (user_id, feedback_text) VALUES (?, ?)", (user_id, feedback))
                self.conn.commit()
                CTkMessagebox(title="Thank You", message="Thank you for your feedback!")

        feedback_label = ctk.CTkLabel(home_screen, text="Would you like to provide feedback on your previous orders?", font=("Arial", 12))
        feedback_label.pack(pady=10)
        feedback_button = ctk.CTkButton(home_screen, text="Provide Feedback", command=feedback_provided)
        feedback_button.pack(pady=10)

        rec2_label = ctk.CTkLabel(home_screen, text="For personalised recommendations click on order food below", font=("Arial", 10))
        rec2_label.pack(pady=8)

        if previous_orders:
            recommendation_label = ctk.CTkLabel(home_screen, text="Based on your previous orders:", font=("Arial", 12))
            recommendation_label.pack(pady=10)

            food_list_frame = ctk.CTkFrame(home_screen)
            food_list_frame.pack(pady=10, fill=ctk.BOTH, expand=True)

            selected_items = []
            previous_orders_list = previous_orders.split(', ')

            for item in previous_orders_list:
                var = ctk.BooleanVar()
                checkbox = ctk.CTkCheckBox(food_list_frame, text=item, variable=var, font=("Helvetica", 12))
                checkbox.pack(anchor="w", padx=20)
                selected_items.append((item, var))

            def remove_items():
                selected_order_items = [item for item, var in selected_items if not var.get()]
                new_orders_str = ', '.join(selected_order_items)

                self.c.execute("UPDATE users SET orders = ? WHERE id = ?", (new_orders_str, user_id))
                self.conn.commit()
                CTkMessagebox(title="Update", message="Orders list updated successfully!")
                home_screen.destroy()
                self.show_home_screen(user_name, user_id)

            remove_button = ctk.CTkButton(home_screen, text="Remove Selected Items", command=remove_items)
            remove_button.pack(pady=10)
        else:
            recommendation_label = ctk.CTkLabel(home_screen, text="No previous orders found.", font=("Arial", 12))
            recommendation_label.pack(pady=10)
            # home_screen.destroy()
            self.current_user_processing = False

        order_button = ctk.CTkButton(home_screen, text="Order Food", command=lambda: self.order_food(user_id, home_screen))
        order_button.pack(pady=10)

        assist_button = ctk.CTkButton(home_screen, text="Need Help?", command=self.assist_with_questions)
        assist_button.pack(pady=10)

    def order_food(self, user_id, home_screen):
        """Allow the user to select food items from previous orders or enter a new one""" 
        food_window = ctk.CTkToplevel(self)
        food_window.title("Order Food")
        food_window.geometry("400x300")
        food_window.config(bg="#f0f0f0")

        # veg_nonveg = simpledialog.askstring("Preference", "Would you like Veg or Non-Veg food?", parent=food_window)

        # if veg_nonveg not in ["Veg", "Non-Veg"]:
        #     messagebox.showerror("Error", "Please choose either Veg or Non-Veg.")
        #     food_window.destroy()
        #     return
        
        self.c.execute("SELECT orders FROM users WHERE id = ?", (user_id,))
        previous_orders = self.c.fetchone()[0]

        # order_label = ctk.CTkLabel(food_window, text=f"Select {veg_nonveg} food items or enter a new one:", font=("Helvetica", 12), bg="#f0f0f0", fg="#333")
        # order_label.pack(pady=10)

        recommendations = self.recommender.generate_recommendations(user_id)
        # rec_label = ctk.CTkLabel(food_window, text=f"Recommended for you: {', '.join(recommendations)}", font=("Arial", 12))
        # rec_label.pack(pady=10)

        self.c.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        user_name = self.c.fetchone()[0]

        selected_items = []

        if previous_orders:
            previous_orders_list = previous_orders.split(', ')

            for item in previous_orders_list:
                var = ctk.BooleanVar()
                checkbox = ctk.CTkCheckBox(food_window, text=item, variable=var, font=("Helvetica", 12), bg="#f0f0f0", fg="#333")
                checkbox.pack(anchor="w", padx=200)
                selected_items.append((item, var))

            if recommendations.empty:
                no_rec_label = ctk.CTkLabel(food_window, text="No recommendations found.", font=("Helvetica", 12), bg="#f0f0f0", fg="#333")
                no_rec_label.pack(pady=10)
                order_now_label = ctk.CTkLabel(food_window, text="To order, click on the button below.", font=("Arial", 10))
                order_now_label.pack(pady=5)
            else:
                for food_name in recommendations:
                    var = ctk.BooleanVar()
                    checkbox = ctk.CTkCheckBox(food_window, text=f"Recommended: {food_name}", variable=var, font=("Helvetica", 12))
                    checkbox.pack(anchor="w", padx=200)
                    selected_items.append((food_name, var))

        new_order_entry = ctk.CTkEntry(food_window, font=("Helvetica", 12))
        new_order_entry.pack(pady=10)

        def place_order():
            new_order = new_order_entry.get()

            if any(var.get() for item, var in selected_items) or new_order:
                selected_order_items = [item for item, var in selected_items if var.get()]

                if new_order:
                    recommendations = self.recommender.generate_recommendations(new_order)
                    for food_name in recommendations:
                        var = ctk.BooleanVar()
                        checkbox = ctk.CTkCheckBox(food_window, text=f"Recommended: {', '.join(recommendations)}", variable=var, font=("Helvetica", 12))
                        checkbox.pack(anchor="w", padx=20)
                        selected_items.append((food_name, var))

                    # messagebox.showinfo("Recommendations", f"Based on your order, we recommend: {', '.join(recommendations)}")
                    # selected_order_items.append(new_order)

                self.c.execute("SELECT orders FROM users WHERE id = ?", (user_id,))
                previous_orders = self.c.fetchone()[0]

                if previous_orders:
                    new_orders_list = previous_orders.split(', ')
                    new_orders_list.extend(selected_order_items)
                    new_orders_str = ', '.join(new_orders_list)
                else:
                    new_orders_str = ', '.join(selected_order_items)

                self.c.execute("UPDATE users SET orders = ? WHERE id = ?", (new_orders_str, user_id))
                self.conn.commit()
                CTkMessagebox(title="Order", message="Order placed successfully!")

                food_window.destroy()
                # self.current_user_processing = False
                self.show_home_screen(user_name, user_id)

                if not self.user_queue.empty():
                    next_user_encoding = self.user_queue.get()
                    self.process_face(next_user_encoding)
                else:
                    self.current_user_processing = False
            else:
                CTkMessagebox(title="Error", message="Please select or enter a food item.")



        place_order_button = ctk.CTkButton(food_window, text="Place Order", command=place_order)
        place_order_button.pack(pady=10)

    def assist_with_questions(self):
        """Provide help or direct the customer to staff"""
        CTkMessagebox(title="Need Help?", message="If you have any questions or need assistance, our staff is here to help!")

    def reload_app(self):
        """Reload the application and cancel the after callback"""
        if self.after_id is not None:
            self.after_cancel(self.after_id) 

        self.destroy()
        RecogniseFaceOrderFood()

if __name__ == "__main__":
    RecogniseFaceOrderFood()





