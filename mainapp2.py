import customtkinter as ctk
import numpy as np
from CTkMessagebox import CTkMessagebox
import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
from PIL import Image, ImageDraw, ImageTk
import face_recognition
import sqlite3
import threading
import queue
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from dateutil import parser
from datetime import datetime

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

class CameraWindow(threading.Thread):  
   def __init__(self, queue):  
      threading.Thread.__init__(self)  
      self.queue = queue  
  
   def run(self):  
      """Continuously capture face from camera feed."""  
      cap = cv2.VideoCapture(0)  
      if not cap.isOpened():  
        self.queue.put("Camera not found!")  
        return  
  
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
              face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]  
              self.queue.put(face_encoding)  
  
              # Smile detection  
              face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)  
              for face_landmarks in face_landmarks_list:  
                left_eye = face_landmarks['left_eye']  
                right_eye = face_landmarks['right_eye']  
                left_eyebrow = face_landmarks['left_eyebrow']  
                right_eyebrow = face_landmarks['right_eyebrow']  
                nose_bridge = face_landmarks['nose_bridge']  
                nose_tip = face_landmarks['nose_tip']  
                top_lip = face_landmarks['top_lip']  
                bottom_lip = face_landmarks['bottom_lip']  
  
                # Calculate the distance between the top and bottom lip  
                lip_distance = abs(top_lip[0][1] - bottom_lip[0][1])  
  
                # If the lip distance is greater than a certain threshold, it's a smile  
                if lip_distance > 10:  
                   cv2.putText(frame, "Smile detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  
                else:  
                   cv2.putText(frame, "Smile, you're on camera!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  
  
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
        img = Image.fromarray(img)   
        imgtk = ImageTk.PhotoImage(image=img)   
    
        # Display the frame in the tkinter label   
        # self.label.configure(image=imgtk)   
        # self.label.image = imgtk  
  
  
        # cv2.imshow("Camera - Face Detection", frame)  
  
        # if cv2.waitKey(1) & 0xFF == ord('q'):  
        #    break  
  
      cap.release()  
      cv2.destroyAllWindows()

class RecogniseFaceOrderFood(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.conn = sqlite3.connect('users.db', check_same_thread=False)
        self.c = self.conn.cursor()
        self.recommender = FoodRecommender(self.c)
        self.queue = queue.Queue()

        self.current_user_processing = False
        self.user_found = False

        self.title("Robot System for Personalized Food Recommendations")
        self.geometry("1000x600")

        
        self.welcome_label = ctk.CTkLabel(self, text="Welcome to GPRobo", fg_color="#FF3C5A",font=("Helvetica", 24, "bold"))
        self.welcome_label.place(rely=0.05,relx=0.5, anchor="center")

        # Initialize database
        self.initialize_database()

        # UI
        self.setup_ui()
 

        # Start camera thread
        self.camera_thread = CameraWindow(self.queue)
        self.camera_thread.start()
        self.user_queue = queue.Queue()

        # Start processing the queue for user recognition
        self.process_queue()

    def initialize_database(self):
        """Initialize the database schema."""
        self.c.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT,
                            email TEXT,
                            face_encoding BLOB,
                            consent INTEGER DEFAULT 0,
                            orders TEXT,
                            dob TEXT)''')
        


        self.conn.commit()

    def setup_ui(self):
        """Set up the main window with frames for switching views."""
        
        self.setup_frame = ctk.CTkFrame(self, fg_color="black", width=700, height=500)
        self.setup_frame.place(relx=0.5, rely=0.5, anchor="center")

               # self.show_home_screen()
        self.user_image = ctk.CTkImage(light_image=Image.open('user_detected.jpg'), size=(400,300))
        self.user_image_label = ctk.CTkLabel(self.setup_frame, text=None, image=self.user_image, fg_color="#FF3C5A",font=("Helvetica", 24, "bold"))
        self.user_image_label.place(rely=0.5,relx=0.5, anchor="center")
        
        self.reco_label = ctk.CTkLabel(self.setup_frame, text="Recognizing Face...", fg_color="green",font=("Helvetica", 24, "bold"))
        self.reco_label.place(relx=0.5, rely=0.7, anchor="center")


    def greet_user(self, consent, user_name, user_id):
        self.main_frame = ctk.CTkFrame(self, fg_color="#0d0d0d", width=1000, height=550)
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center")


        refresh_button = ctk.CTkButton(self.main_frame, text="Refresh Screen", command=lambda: self.refresh_screen(consent, user_name, user_id))  
        refresh_button.place(relx=0.5, rely=0.5, anchor="center")

        # Top Label "User Profile"
        profile_label = ctk.CTkLabel(self.main_frame, text="Your Profile", font=("Helvetica", 24, "bold"), pady=10)
        profile_label.place(relx=0.5, rely=0.05, anchor="center")

        # Label for "Welcome to the restaurant"
        welcome_restaurant_label = ctk.CTkLabel(self.main_frame, text="Welcome to the restaurant", text_color="#FFFFFF", font=("Helvetica", 24, "bold"), pady=10)
        welcome_restaurant_label.place(relx=0.5, rely=0.4, anchor="center")
        
        if consent == 1:
            try:
                rounded_image = self.create_rounded_image('user_detected.jpg', size=(150, 100))
            except FileNotFoundError:
                rounded_image = self.create_rounded_image('default_placeholder.jpg', size=(100, 100))
            
            self.c.execute("SELECT dob FROM users WHERE id = ?", (user_id,))
            user_dob = self.c.fetchone()[0]
            specialday_msg = ""

            profile_message = f"Hello \n{user_name}"
            profile_message_label = ctk.CTkLabel(self.main_frame, text=profile_message, text_color="#FFFFFF", font=("Roboto", 18, "bold"))
            profile_message_label.place(relx=0.5, rely=0.2, anchor="center")

            welcome_message = f"{user_name}"


            if user_dob:
                user_dob = parser.parse(user_dob)
                today = datetime.today() 

                if user_dob.month == today.month and user_dob.day == today.day:
                    specialday_msg = f"Happy Birthday!, {user_name}! We've got a surprise \ndessert waiting for you!"
                else:
                    return None

            welcome_label = ctk.CTkLabel(self.main_frame, text=welcome_message, text_color="#FFFFFF", font=("Roboto", 18, "bold"))
            welcome_label.place(relx=0.1, rely=0.35, anchor="center")

            specialday_label = ctk.CTkLabel(self.main_frame, text=specialday_msg, font=("Roboto", 14, "italic"))
            specialday_label.place(relx=0.8, rely=0.3, anchor="center")

            image_frame = ctk.CTkFrame(self.main_frame, fg_color="#262626", corner_radius=50, width=140, height=140)
            image_frame.place(relx=0.1, rely=0.2, anchor="center")

            rounded_image_tk = ctk.CTkImage(light_image=rounded_image, size=(150, 150))
            self.profile_image_label = ctk.CTkLabel(image_frame, text=None, fg_color="#262626", bg_color="#144870", image=rounded_image_tk)
            self.profile_image_label.place(relx=0.5, rely=0.5, anchor="center")

            # Fetch and display previous orders
            self.prev_frame = ctk.CTkScrollableFrame(self.main_frame, width=200, height=200)
            self.prev_frame.place(relx=0.85, rely=0.75, anchor="center")

            self.order_frame = ctk.CTkScrollableFrame(self.main_frame, width=200, height=200)
            self.order_frame.place(relx=0.15, rely=0.75, anchor="center")

            previous_orders_label = ctk.CTkLabel(self.main_frame, text="Your Previous Orders:", font=("Helvetica", 12, "bold"))  
            previous_orders_label.place(relx=0.85, rely=0.5, anchor="center")
            recommendation_text = "Your Recommendations will display here"
            recommended_orders_label = ctk.CTkLabel(self.main_frame, text=recommendation_text, font=("Helvetica", 12, "bold"))  
            recommended_orders_label.place(relx=0.15, rely=0.5, anchor="center") 

            self.c.execute("SELECT orders FROM users WHERE id = ?", (user_id,))
            previous_orders = self.c.fetchone()[0]

            if previous_orders:
                previous_orders_list = previous_orders.split(",")
                for order in previous_orders_list:
                    checkbox = ctk.CTkCheckBox(self.prev_frame, text=order.strip(), font=("Helvetica", 12))
                    checkbox.pack(anchor="w", padx=20)
            else:
                recommendation_label = ctk.CTkLabel(self.prev_frame, text="No previous orders found.", font=("Helvetica", 12))
                recommendation_label.pack(pady=10)

            # Button to place a new order
            self.back_button = ctk.CTkButton(self.main_frame, text="Order Food", command=lambda: self.show_recommendations(consent, user_name, user_id))
            self.back_button.place(relx=0.5, rely=0.7, anchor="center")

        else:
            self.ask_for_consent(user_name, user_id)
            

    def show_recommendations(self, consent, user_name, user_id):
        """Show the food recommendations screen."""
        self.rec_frame = ctk.CTkFrame(self.main_frame, fg_color=("#262626"), width=300 , height=200)
        self.rec_frame.place(relx=0.5, rely=0.75, anchor="center")
            # self.clear_frame()
        self.rec_label = ctk.CTkLabel(self.rec_frame, text="Enter a dish name:")
        self.rec_label.place(relx=0.5, rely=0.1, anchor="center")

        self.rec_entry = ctk.CTkEntry(self.rec_frame)
        self.rec_entry.place(relx=0.5, rely=0.25, anchor="center")

        self.rec_button = ctk.CTkButton(self.rec_frame, text="Place Order", command= lambda: self.get_recommendations(user_id))
        self.rec_button.place(relx=0.5, rely=0.5, anchor="center")

        self.back_button = ctk.CTkButton(self.rec_frame, text="Close", command=self.rec_frame.place_forget)
        self.back_button.place(relx=0.5, rely=0.9, anchor="center")

            # To display the recommendations frame
        # self.order_screen_frame = ctk.CTkFrame(self, fg_color="yellow", width=200, height=200)
        # self.order_screen_frame.place(relx=0.7, rely=0.7, anchor="center")
        # self.order_screen_close = ctk.CTkButton(self.order_screen_frame, text="Close", command=self.order_screen_frame.place_forget)
        # self.order_screen_close.place(relx=0.5, rely=0.9, anchor="center")
        
    def clear_frame(self):
        """Clear the current frame before switching views."""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def order_food(self):
        """Handle ordering food."""
        food_name = self.rec_entry.get()
        if food_name:
            messagebox.showinfo("Order", f"You have ordered {food_name}.")
            # Here you can also save the order in the database


    def get_recommendations(self, user_id):  
        """Handle generating recommendations based on the user's input."""  
        food_name = self.rec_entry.get()  
        
        selected_items = []  
        self.selected_checkboxes = []

        for widget in self.prev_frame.winfo_children():
            widget.destroy()

        # Fetch previous orders from the database  
        self.c.execute("SELECT orders FROM users WHERE id = ?", (user_id,))  
        result = self.c.fetchone()  
        
        if result and result[0]:  
            previous_orders = result[0]   
            print("Previous orders fetched from DB:", "".join(previous_orders.split(",")))   
        
            # Display previous orders as checkboxes  
            previous_orders_list = previous_orders.split(",")   
            # previous_orders_label = ctk.CTkLabel(self.main_frame, text="Your Previous Orders:", font=("Helvetica", 12))  
            # previous_orders_label.place(relx=0.85, rely=0.25, anchor="center")
            for order in previous_orders_list:  
                var = ctk.BooleanVar()  
                checkbox = ctk.CTkCheckBox(self.prev_frame, text=order.strip(), font=("Helvetica", 12), variable=var)  
                checkbox.pack(anchor="w", padx=20)  
                self.selected_checkboxes.append((checkbox, var))   
        else:  
            print("No previous orders found for user:", user_id)   
        
        # If a food name is entered then generate recommendations  
        if food_name:  
            recommended_foods = self.recommender.generate_recommendations(food_name)  
            
            if recommended_foods:  
                recommendation_text = "\nRecommended: " + ", ".join(recommended_foods)  
        
                selected_items.append(food_name)  
                selected_items.extend(recommended_foods)  
                
                # recommended_orders_label = ctk.CTkLabel(self.order_frame, text="Recommended Orders:", font=("Helvetica", 12))  
                # recommended_orders_label.pack(anchor="w", padx=20)  
                for food in selected_items:  
                    var = ctk.BooleanVar()   
                    checkbox = ctk.CTkCheckBox(self.order_frame, text=food.strip(), font=("Helvetica", 12), variable=var)  
                    checkbox.pack(anchor="w", padx=20)  
                    self.selected_checkboxes.append((checkbox, var))   
                    print("Food name:", food)  
            else:  
                CTkMessagebox(title="No Match", message="We couldn't find any similar dishes.")  
        else:  
            CTkMessagebox(title="Input Error", message="Please enter a food name.")  
        
        save_button = ctk.CTkButton(self.order_frame, text="Save Orders", command=lambda: self.save_orders(user_id))   
        save_button.pack(pady=10)

        remove_button = ctk.CTkButton(self.prev_frame, text="Remove Orders", command=lambda: self.remove_orders(user_id))   
        remove_button.pack(pady=10)

    # Once recommednations are generated user can add orders to the list
    def save_orders(self, user_id):
        """Save the selected orders to the database."""
        selected_items = []

        for checkbox, var in self.selected_checkboxes:
            if var.get():  
                item_text = checkbox.cget("text")
          
                if item_text.startswith("Recommended: "):
                    item_text = item_text.replace("Recommended: ", "")
                
                selected_items.append(item_text) 

        self.c.execute("SELECT orders FROM users WHERE id = ?", (user_id,))
        result = self.c.fetchone()
        previous_orders = result[0] if result else None

        if previous_orders:
            previous_orders_list = previous_orders.split(", ")  
            all_orders = previous_orders_list + selected_items
        else:
            all_orders = selected_items

        new_orders_str = ', '.join(all_orders) 

        self.c.execute("UPDATE users SET orders = ? WHERE id = ?", (new_orders_str, user_id))
        self.conn.commit()

        CTkMessagebox(title="Order", message="Order placed successfully!")

        # Dubgging the output 
        print("Selected items:", selected_items)
        print("Previous orders:", previous_orders)
        print("All orders:", all_orders)


    def remove_orders(self, user_id):  
        """Remove the selected orders from the database."""  
        orders_to_remove = []  
        for checkbox, var in self.selected_checkboxes:  
            if var.get():  
                orders_to_remove.append(checkbox.cget("text"))  

        self.c.execute("SELECT orders FROM users WHERE id = ?", (user_id,))
        existing_orders = self.c.fetchone()[0]

        if existing_orders:
            current_orders = existing_orders.split(",")

            updated_orders = [order for order in current_orders if order not in orders_to_remove]

            self.c.execute("UPDATE users SET orders = ? WHERE id = ?", (",".join(updated_orders), user_id))
            self.conn.commit()

            CTkMessagebox(title="Orders Removed", message="Selected orders have been removed successfully.")
        else:
            CTkMessagebox(title="No Orders Found", message="No orders found for this user.")


    def process_queue(self):
        """Process the queue for messages or face encodings."""
        try:
            encoding = self.queue.get_nowait()
            if isinstance(encoding, np.ndarray):
                self.process_face(encoding)
        except queue.Empty:
            self.after(100, self.process_queue)


    def ask_for_consent(self, user_name, user_id):
        """Ask the user for consent to store and use their data for personalized recommendations."""
        consent_window = ctk.CTkToplevel(self)
        consent_window.title("Consent Form")
        consent_window.geometry("400x200")

        consent_text = ctk.CTkLabel(consent_window, text=f"Hi {user_name}, do you consent to personalized food recommendations?")
        consent_text.pack(pady=10)

        def handle_consent(choice):
            if choice == "Yes":
                self.c.execute("UPDATE users SET consent = 1 WHERE id = ?", (user_id,))
                self.conn.commit()
                CTkMessagebox(title="Consent", message="Thank you for consenting!")
                # self.show_home_screen(user_name, user_id)
            else:
                CTkMessagebox(title="Notice", message="You can still enjoy our services without personalized recommendations.")
            consent_window.destroy()

        yes_button = ctk.CTkButton(consent_window, text="Yes", command=lambda: handle_consent("Yes"))
        no_button = ctk.CTkButton(consent_window, text="No", command=lambda: handle_consent("No"))

        yes_button.pack(side=tk.LEFT, padx=20, pady=20)
        no_button.pack(side=tk.RIGHT, padx=20, pady=20)




    # def remove_order(self, user_id, order_to_remove):
    #     """Remove a specific order from the previous orders."""
    #     self.c.execute("SELECT orders FROM users WHERE id = ?", (user_id,))
    #     previous_orders = self.c.fetchone()[0]

    #     if previous_orders:
    #         previous_orders_list = previous_orders.split(",")
    #         if order_to_remove in previous_orders_list:
    #             previous_orders_list.remove(order_to_remove)
    #             new_orders = ",".join(previous_orders_list)
    #             self.c.execute("UPDATE users SET orders = ? WHERE id = ?", (new_orders, user_id))
    #             self.conn.commit()

    #             CTkMessagebox(title="Removed", message=f"{order_to_remove} has been removed from your orders.")
    #             self.greet_user(user_id, 1, user_id)]


    def process_face(self, encoding):
        """Process the detected face (login or register)"""
        def callback():
            self.reco_label.pack_forget()
            self.c.execute("SELECT id, name, face_encoding, consent FROM users")
            users = self.c.fetchall()

            user_found = False

            for user in users:
                user_id, user_name, user_encoding, consent = user
                known_encoding = np.frombuffer(user_encoding, dtype=np.float64)

                matches = face_recognition.compare_faces([known_encoding], encoding)
                if matches[0]:
                    user_found = True
                    self.greet_user(consent, user_name, user_id)
                    return

            if not user_found:
                self.register_user(encoding, user_id)
    
        self.after(0, callback)


    # def process_face(self, encoding):
    #     """Process the detected face by checking if it's a known user."""
    #     self.c.execute("SELECT * FROM users")
    #     user_encoding = self.c.fetchall()

    #     for user in user_encoding:
    #         # user_id, user_name, user_encoding, consent, orders, dob = user
    #         user_id, user_name, email, user_encoding, consent, orders, dob = user 
    #         known_encoding = np.frombuffer(user_encoding, dtype=np.float64)
    #         print(known_encoding)

    #         matches = face_recognition.compare_faces([known_encoding], encoding)
    #         if matches[0]:
    #             user_found = True
    #             self.greet_user(consent, user_name, user_id)
                
    #             if self.reco_label:
    #                 self.reco_label.destroy()

    #             if self.user_image_label:
    #                 self.user_image_label.destroy()
    #             return 


    #     # if not user_found:
    #     #     self.register_user(encoding)
    #     if not self.current_user_processing:
    #         self.current_user_processing = True
    #         self.register_user(encoding)


    def register_user(self, encoding, user_id):
        """Register a new user if the face is not recognized."""
        self.register_frame = ctk.CTkFrame(self, fg_color="#0d0d0d", width=800, height=500)
        self.register_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.name_label = ctk.CTkLabel(self.register_frame, text="Name")
        self.name_label.place(relx=0.1, rely=0.1, anchor="center")
        self.name_entry = ctk.CTkEntry(self.register_frame)
        self.name_entry.place(relx=0.1, rely=0.2, anchor="center")

        self.email_label = ctk.CTkLabel(self.register_frame, text="Email")
        self.email_label.place(relx=0.1, rely=0.3, anchor="center")
        self.email_entry = ctk.CTkEntry(self.register_frame)
        self.email_entry.place(relx=0.1, rely=0.4, anchor="center")

        self.dob_label = ctk.CTkLabel(self.register_frame, text="Date of Birth")
        self.dob_label.place(relx=0.1, rely=0.5, anchor="center")
        self.dob_entry = ctk.CTkEntry(self.register_frame)
        self.dob_entry.place(relx=0.1, rely=0.6, anchor="center")

        register_button = ctk.CTkButton(self.register_frame, text="Register", command= self.complete_registration)
        register_button.place(relx=0.1, rely=0.7, anchor="center")

        self.encoding = encoding
        

        # name = simpledialog.askstring("Input", "Please enter your name:", parent=self)
        # email = simpledialog.askstring("Input", "Please enter your email:", parent=self)
        # dob = simpledialog.askstring("Input", "Please enter your date of birth :", parent=self)

        # try:
        #     parsed_dob = parser.parse(dob)
        # except ValueError:
        #     messagebox.showerror("Error", "Invalid date format. Please enter in YYYY-MM-DD format.")
        #     return

        # encoding_blob = encoding.tobytes()
        # self.c.execute("INSERT INTO users (name, email, face_encoding, consent, dob) VALUES (?, ?, ?, 1, ?)", (self.name_entry.get(), self.email_entry.get(), encoding_blob, self.dob_entry.get()))
        # self.conn.commit()
        # CTkMessagebox(title="Registration", message="You have been successfully registered!")

        # encoding_blob = encoding.tobytes()
        # self.c.execute("INSERT INTO users (name, email, face_encoding, consent, dob) VALUES (?, ?, ?, 1, ?)", (name, email, encoding_blob, dob))
        # self.conn.commit()
        # CTkMessagebox(title="Registration", message="You have been successfully registered!")
        # self.show_home_screen(name, self.c.lastrowid)


    def complete_registration(self):  
        """Complete the registration process after the user has entered their details."""  
        name = self.name_entry.get()  
        email = self.email_entry.get()  
        dob = self.dob_entry.get()  
            
        # try:  
        #     parsed_dob = parser.parse(dob)  
        # except ValueError:  
        #     messagebox.showerror("Error", "Invalid date format. Please enter in YYYY-MM-DD format.")  
        #     return  
            
        encoding_blob = self.encoding.tobytes()  
        self.c.execute("INSERT INTO users (name, email, face_encoding, consent, dob) VALUES (?, ?, ?, 1, ?)", (name, email, encoding_blob, dob))  
        self.conn.commit()  
        CTkMessagebox(title="Registration", message="You have been successfully registered!")  
        self.register_frame.destroy()  
        # self.greet_user(1, name, self.c.lastrowid)

    def create_rounded_image(self,image_path, size=(150, 100)):
            img = Image.open(image_path).resize((150, 170), Image.LANCZOS).convert("RGBA")
            rounded_rect = Image.new('RGBA', (180, 220), (0, 0, 0, 0))
            draw = ImageDraw.Draw(rounded_rect)
            bg_color = (50, 50, 50)
            draw.rounded_rectangle([0, 0, 180, 220], radius=50, fill=bg_color)
            img_position = (15, 25)  # Adjust position as needed
            rounded_rect.paste(img, img_position, img)
            border_radius = 10  
            draw.rounded_rectangle([img_position[0], img_position[1], img_position[0] + img.size[0], img_position[1] + img.size[1]], 
            radius=border_radius, outline=(255, 255, 255), width=2)  
            return rounded_rect

    
    def refresh_screen(self, consent, user_name, user_id):  
        """Refresh the screen by clearing the current frame."""  
        self.clear_frame()  
        self.greet_user(consent, user_name, user_id)
    

if __name__ == "__main__":
    app = RecogniseFaceOrderFood()
    app.mainloop()