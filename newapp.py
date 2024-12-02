import customtkinter as ctk
import sqlite3
import cv2
from tkinter import messagebox
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Initialize database connection
conn = sqlite3.connect('customers.db')
cursor = conn.cursor()

# Create tables if not exist
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, email TEXT, dob TEXT, consent INTEGER)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER, food_item TEXT, 
    FOREIGN KEY(user_id) REFERENCES users(user_id))''')

# To simulate recommendations (you can replace this with your logic)
cursor.execute('''CREATE TABLE IF NOT EXISTS recommendations (
    rec_id INTEGER PRIMARY KEY AUTOINCREMENT,
    food_item TEXT)''')

conn.commit()

class FoodRecommender:
    def __init__(self, cursor):
        self.c = cursor
        self.load_food_data()

    def load_food_data(self):
        """Load and process the food data for recommendations."""
        self.food_data = pd.read_csv("indian_food.csv")
        self.food_df = pd.DataFrame(self.food_data, columns=["Name", "Ingredients", "Veg_Non", "Flavour", "Course"])

        # Preprocess ingredients and compute similarity
        self.food_df['Ingredients'] = self.food_df['Ingredients'].str.split(',').apply(lambda x: [i.strip() for i in x])
        mlb = pd.get_dummies(self.food_df['Ingredients'].apply(pd.Series).stack()).groupby(level=0).sum()
        self.final_df = pd.concat([self.food_df['Name'], mlb], axis=1)
        self.cosine_sim = cosine_similarity(self.final_df.drop(columns=['Name']))

    def generate_recommendations(self, food_name):
        """Generate food recommendations based on cosine similarity."""
        indices = pd.Series(self.final_df.index, index=self.final_df['Name']).drop_duplicates()
        if food_name not in indices:
            return []  # No match found

        idx = indices[food_name]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        dish_indices = [i[0] for i in sim_scores]
        return self.final_df['Name'].iloc[dish_indices].tolist()

class FoodOrderingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("600x400")
        self.title("Food Ordering App")

        # Frames for ordering and recommendations
        self.order_frame = ctk.CTkFrame(self, width=300, height=400)
        self.recommend_frame = ctk.CTkFrame(self, width=300, height=400)
        
        self.order_frame.place(relx=0.1, rely=0.1)
        self.recommend_frame.place(relx=0.6, rely=0.1)

        self.detect_face_active = True

        # Greeting method to check user existence
        self.check_user()

        # Face recognition setup
        self.cap = cv2.VideoCapture(0)
        self.detect_face()

    def check_user(self):
        # Check for existing users
        existing_users = cursor.execute("SELECT * FROM users").fetchall()
        if existing_users:
            self.greet_user(existing_users[0][1])  # Greet the first existing user
        else:
            self.show_consent_form()  # Show consent form for new users

    def greet_user(self, user_name):
        # Greet the returning user
        greeting_label = ctk.CTkLabel(self.order_frame, text=f"Welcome back, {user_name}!")
        greeting_label.place(relx=0.5, rely=0.1, anchor='center')

        # Show home message
        home_label = ctk.CTkLabel(self.order_frame, text="Personalized food recommendations system")
        home_label.place(relx=0.5, rely=0.3, anchor='center')

        # Load previous orders
        self.load_food_ordering()

    def show_consent_form(self):
        # Clear existing widgets from the frame
        for widget in self.order_frame.winfo_children():
            widget.destroy()

        # Now add the input fields and checkbox
        self.name_entry = ctk.CTkEntry(self.order_frame, placeholder_text="Enter your name")
        self.name_entry.place(relx=0.5, rely=0.2, anchor='center')

        self.email_entry = ctk.CTkEntry(self.order_frame, placeholder_text="Enter your email")
        self.email_entry.place(relx=0.5, rely=0.3, anchor='center')

        self.dob_entry = ctk.CTkEntry(self.order_frame, placeholder_text="Enter your DOB")
        self.dob_entry.place(relx=0.5, rely=0.4, anchor='center')

        self.consent_var = ctk.BooleanVar()
        self.consent_checkbox = ctk.CTkCheckBox(self.order_frame, text="I agree to provide my details",
                                                variable=self.consent_var)
        self.consent_checkbox.place(relx=0.5, rely=0.5, anchor='center')

        self.submit_btn = ctk.CTkButton(self.order_frame, text="Submit", command=self.process_consent)
        self.submit_btn.place(relx=0.5, rely=0.6, anchor='center')

    def process_consent(self):
        # Process user's consent and save details in database
        if self.consent_var.get():
            name = self.name_entry.get()
            email = self.email_entry.get()
            dob = self.dob_entry.get()

            # Save to users table
            cursor.execute("INSERT INTO users (name, email, dob, consent) VALUES (?, ?, ?, ?)",
                           (name, email, dob, 1))
            conn.commit()

            messagebox.showinfo("Success", "Consent granted. Proceeding with food ordering.")
            self.detect_face_active = False
            self.load_food_ordering()

        else:
            messagebox.showwarning("Warning", "Consent not granted. Returning to home screen.")
            self.reset_form()

    def reset_form(self):
        # Reset form if consent is not granted
        self.name_entry.delete(0, 'end')
        self.email_entry.delete(0, 'end')
        self.dob_entry.delete(0, 'end')
        self.consent_var.set(0)

    def load_food_ordering(self):
        # After consent, load the food ordering interface
        self.clear_frames()

        # Display previously ordered food
        user_orders = cursor.execute("SELECT food_item FROM orders WHERE user_id = (SELECT MAX(user_id) FROM users)").fetchall()
        ctk.CTkLabel(self.order_frame, text="Your Previous Orders:").place(relx=0.5, rely=0.1, anchor='center')
        for i, order in enumerate(user_orders):
            ctk.CTkLabel(self.order_frame, text=order[0]).place(relx=0.5, rely=0.2 + 0.05*i, anchor='center')

        # Checkbox to add new food
        self.food_var = ctk.StringVar()
        self.food_entry = ctk.CTkEntry(self.order_frame, placeholder_text="Enter food item")
        self.food_entry.place(relx=0.5, rely=0.5 + 0.05*len(user_orders), anchor='center')

        self.add_food_btn = ctk.CTkButton(self.order_frame, text="Add Food", command=self.add_food)
        self.add_food_btn.place(relx=0.5, rely=0.6 + 0.05*len(user_orders), anchor='center')

        # Display recommended food
        rec_foods = cursor.execute("SELECT food_item FROM recommendations").fetchall()
        ctk.CTkLabel(self.recommend_frame, text="Recommended Foods:").place(relx=0.5, rely=0.1, anchor='center')
        for i, food in enumerate(rec_foods):
            ctk.CTkLabel(self.recommend_frame, text=food[0]).place(relx=0.5, rely=0.2 + 0.05*i, anchor='center')

    def add_food(self):
        # Add food to user's orders
        food_item = self.food_entry.get()
        cursor.execute("INSERT INTO orders (user_id, food_item) VALUES ((SELECT MAX(user_id) FROM users), ?)", (food_item,))
        conn.commit()
        messagebox.showinfo("Success", f"Added {food_item} to your orders.")
        self.load_food_ordering()

    def detect_face(self):
        # Face recognition logic (using OpenCV in the background)
        def face_recognition():
            if not self.detect_face_active:
                return  # Stop face detection if user is registered or logged in

            _, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                # If face detected, show the consent form once
                if not hasattr(self, 'consent_shown') or not self.consent_shown:
                    self.show_consent_form()
                    self.consent_shown = True  # Flag to show form only once
            else:
                self.consent_shown = False  # Reset when no face detected

            self.after(100, face_recognition)  # Continue detection

        face_recognition()

    def clear_frames(self):
        # Clear frame contents for switching views
        for widget in self.order_frame.winfo_children():
            widget.destroy()
        for widget in self.recommend_frame.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    app = FoodOrderingApp()
    app.mainloop()
