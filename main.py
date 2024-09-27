import joblib
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd

# Load scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('mlmodel.pkl')


def predict():
    try:
        average_cost = float(avg_cost_entry.get())
        table_booking = table_booking_var.get()
        table_booking_status = 1 if table_booking == 'Yes' else 0
        online_delivery = online_delivery_var.get()
        online_delivery_status = 1 if online_delivery == 'Yes' else 0
        price_range = int(price_range_var.get())

        # Prepare input data as a DataFrame with the correct column names
        input_data = pd.DataFrame({
            'Average Cost for two': [average_cost],
            'Has Table booking': [table_booking_status],
            'Has Online delivery': [online_delivery_status],
            'Price range': [price_range]
        })

        # Transform the input data using the loaded scaler
        scaled_input = scaler.transform(input_data)

        # Predict using the model
        prediction = model.predict(scaled_input)[0]
        print(prediction)


        if prediction >= 2 and prediction <= 2.5:
            review_text = "Poor"
        elif prediction >= 2.5 and prediction <= 3.5:
            review_text = "Average"
        elif prediction >= 3.5 and prediction <= 4:
            review_text = "Good"
        elif prediction >= 4 and prediction <= 4.5:
            review_text = "Very Good"
        elif prediction >= 4.5 and prediction <= 5:
            review_text = "Excellent"
        else:
            review_text = "Invalid Prediction"

        review_label.config(text=review_text)

    except ValueError as e:
        review_label.config(text=f"Error: {str(e)}")


# Create the main window
root = tk.Tk()
root.title("Restaurant Rating Prediction")

# Create input fields
tk.Label(root, text="Average Cost for two:").grid(row=0, column=0)
avg_cost_entry = tk.Entry(root)
avg_cost_entry.grid(row=0, column=1)

tk.Label(root, text="Table Booking:").grid(row=1, column=0)
table_booking_var = tk.StringVar()
table_booking_combo = ttk.Combobox(root, textvariable=table_booking_var, values=["Yes", "No"])
table_booking_combo.grid(row=1, column=1)

tk.Label(root, text="Online Delivery:").grid(row=2, column=0)
online_delivery_var = tk.StringVar()
online_delivery_combo = ttk.Combobox(root, textvariable=online_delivery_var, values=["Yes", "No"])
online_delivery_combo.grid(row=2, column=1)

tk.Label(root, text="Price Range:").grid(row=3, column=0)
price_range_var = tk.StringVar()
price_range_combo = ttk.Combobox(root, textvariable=price_range_var, values=[1, 2, 3, 4, 5])
price_range_combo.grid(row=3, column=1)

# Create predict button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=4, column=0, columnspan=2)

# Create label to display the review text
review_label = tk.Label(root, text="")
review_label.grid(row=5, column=0, columnspan=2)

# Start the main loop
root.mainloop()
