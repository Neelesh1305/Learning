import streamlit as st
import time
import numpy as np
import ast
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Define the path where the file is saved
original_file_path = "/Users/saineeleshkuntimala/Downloads/Questions1.csv"
# Load the CSV file into a DataFrame
df = pd.read_csv(original_file_path)
df['Received answer'] = None  # Initialize an empty 'Received answer' column
df1 = df.copy()

def encode_unique_elements(df, column_name):
    # Define the mapping dictionary
    mapping = {
        'Prime number': 0,
        'Perfect square': 1,
        'Product of two distinct prime numbers': 2,
        'None of the above': 3
    }
    # Apply the mapping to the specified column
    df[column_name] = df[column_name].map(mapping)
    return df

# Define the session state
ss = st.session_state
if 'counter' not in ss:
    ss.counter = 0
if 'answers' not in ss:
    ss.answers = {}
if 'answers_list' not in ss:
    ss.answers_list = [None] * 20
if 'm' not in ss:
    ss.m = [0]
if 'Grade' not in ss:
    ss.Grade = 0
if 'start_time' not in ss:
    ss.start_time = time.time()
if 'submitted' not in ss:
    ss.submitted = False
if 'selected_option' not in ss:
    ss.selected_option = None

# Define the button click handler function
def btn_click():
    # Ensure the counter is within bounds
    if ss.counter < 20:
        # Get the selected answer
        if ss.selected_option is not None:
            selected_answer = ss.selected_option
            # Store the answer in the session state
            ss.answers[f"Question {ss.counter + 1}"] = selected_answer
            # Also store the answer in the list
            ss.answers_list[ss.counter] = selected_answer
            # Increment the counter only if it's within bounds
            if ss.counter < 20 - 1:
                ss.counter += 1
            # Update the question index list
            update_question_index()
            # Reset selected option
            ss.selected_option = None
        else:
            st.error("Please select an option before moving to the next question.")

# Define the previous button click handler function
def prev_click():
    if ss.counter > 0 and len(ss.m) > 0:
        ss.counter -= 1
        ss.m = ss.m[:-1] 

def update_question_index():
    df1 = df.iloc[ss.m]
    df1["Received answer"].iloc[len(ss.m)-1] = ss.answers_list[len(ss.m)-1]
    label_encoder = LabelEncoder()
    df1 = encode_unique_elements(df1, 'Correct answer')
    df1 = encode_unique_elements(df1, 'Received answer')
    df1['Grade'] = df1['Received answer'] - df1['Correct answer']
    df1['Grade'] = df1['Grade'].replace(0, pd.NA)
    df1['Grade'] = df1['Grade'].replace([-3, -2, -1, 1, 2, 3], 10)
    df1['Grade'].fillna(1, inplace=True)
    df1['Grade'] = df1['Grade'].replace(10, 0)
    dfi = df1[['Difficulty', 'Grade']]
    X = dfi['Difficulty']
    y = dfi['Grade']
    X = X.to_numpy().reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = []
    for j in range(ss.m[-1], ss.m[-1] + 5):
        X_test = [j + 1]
        X_test = np.array(X_test).reshape(-1, 1)
        yk = model.predict(X_test)
        y_pred.append(yk[0])
    max_value = float('-inf')  # Initialize max_value to negative infinity
    max_index = -1  # Initialize max_index to -1
    for k, num in enumerate(y_pred):
        if num >= max_value:
            max_value = num
            max_index = k
    ss.m.append(max_index + ss.m[-1] + 1)
    # Calculate and update the sum of grades
    ss.Grade += dfi['Grade'].iloc[-1].sum()

def submit_test():
    ss.submitted = True
    st.write("End of Quiz")
    st.write("Questions and Answers:", ss.answers)
    st.write("Grade:", ss.Grade, "/20.0")  # Display the total grade at the end

# Timer logic
elapsed_time = time.time() - ss.start_time
remaining_time = 60 - elapsed_time

if remaining_time <= 0:
    submit_test()
else:
    minutes, seconds = divmod(remaining_time, 60)
    st.write(f"Time remaining: {int(minutes):02d}:{int(seconds):02d}")

# Display the current element on the page
if not ss.submitted:
    if ss.counter < 20 and len(ss.m) < 21:
        st.write(len(ss.m))
        st.write(ss.counter)
        df1 = df.iloc[ss.m]
        question_placeholder = st.empty()
        options_placeholder = st.empty()
        # Write the question
        question_placeholder.write(f"**Question {ss.counter + 1} : {df['Questions'].iloc[ss.m[-1]]}**")
        # Get the options
        options_str = df1["Options"].iloc[len(ss.m)-1]
        options = ast.literal_eval(options_str)
        # Initialize the session state for the current question
        if f"Q{ss.counter}" not in st.session_state:
            st.session_state[f"Q{ss.counter}"] = None
        # Display the radio button without the on_change parameter
        selected_option = options_placeholder.radio("", options, index=0 if st.session_state[f"Q{ss.counter}"] is None else options.index(st.session_state[f"Q{ss.counter}"]), key=f"Q{ss.counter}")
        ss.selected_option = selected_option
        
        # Display the "Previous Question" button
        if ss.counter > 0:
            st.button("Previous Question", on_click=prev_click)
        # Display the "Next Question" button
        st.button("Next Question", on_click=btn_click)
        # Update the 'Received answer' in the DataFrame
        if ss.counter < 20:
            st.write(df1)
            st.write(ss.m)
    else:
        st.button("Submit test", on_click=submit_test)
else:
    st.button("Previous Question", on_click=prev_click)
