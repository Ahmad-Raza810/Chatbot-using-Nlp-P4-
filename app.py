import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup SSL and NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot response function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Initialize session state for navigation
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    if st.sidebar.button("🏠 Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("💬 Chatbot"):
        st.session_state.page = "Chatbot"
    if st.sidebar.button("📜 History"):
        st.session_state.page = "History"
    if st.sidebar.button("ℹ️ About"):
        st.session_state.page = "About"
    if st.sidebar.button("🌐 Social Media"):
        st.session_state.page = "Social Media"

    # Navigation logic
    if st.session_state.page == "Home":
        st.title("Welcome to the AI Chatbot! 🤖")
        st.write("""
        Hello! This is your interactive AI chatbot assistant. 🎉
        
        **How to Start Chatting:**
        - Navigate to the **Chatbot** section from the sidebar.
        - Type your message and let the chatbot assist you!
        
        Enjoy exploring the other sections like **History** to view past conversations or **About** to learn more about the project. 😊
        """)

    elif st.session_state.page == "Chatbot":
        st.title("💬 Chat with the Bot")
        st.markdown("""
        <style>
        .chat-icon {
            display: inline-block;
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 50%;
            margin-right: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.write("Type your message below and let’s get started!")

        # Check if the chat log exists
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        # Chat interface
        user_input = st.text_input("👤 You:", placeholder="Type your message here...")
        if user_input:
            response = chatbot(user_input)
            st.markdown(f"<span class='chat-icon'>🤖</span> **Chatbot:** {response}", unsafe_allow_html=True)

            # Save chat log
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            # Goodbye message
            if response.lower() in ['goodbye', 'bye']:
                st.info("👋 Thank you for chatting! Have a great day!")
                st.stop()

    elif st.session_state.page == "History":
        st.title("📜 Conversation History")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    with st.expander(f"🕒 {row[2]}"):
                        st.markdown(f"**👤 User:** {row[0]}")
                        st.markdown(f"**🤖 Chatbot:** {row[1]}")
        except FileNotFoundError:
            st.error("🚫 No conversation history found!")

    elif st.session_state.page == "About":
        st.title("ℹ️ About the Chatbot Project")
        st.write("""
        This chatbot uses **NLP** and **Logistic Regression** to classify intents and respond intelligently. 🎉
        
        **Documentation Section:**
        - **Technologies Used:** Python, Streamlit, NLTK, scikit-learn.
        - **Training:** Utilized TfidfVectorizer and Logistic Regression for intent classification.
        - **Dataset:** Custom JSON file containing intents and patterns.
        - **Features:**
          - Real-time chatting with conversational logging.
          - Easy navigation with a user-friendly interface.
          - Expandable for future features like sentiment analysis.
        
        For more details, explore the project documentation and source code!
        """)

    elif st.session_state.page == "Social Media":
      st.title("🌐 Connect with Me")
      st.write("Follow me on my social platforms:")

      # HTML for unique, colorful icons with hover effect
      social_media_html = """
      <style>
        .social-icons {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }
        .social-icons a {
            text-decoration: none;
            color: inherit;
            transition: transform 0.3s, color 0.3s;
        }
        .social-icons a:hover {
            transform: scale(1.2);
            color: #0073e6; /* Light Blue Hover Effect */
        }
        .social-icons img {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            transition: box-shadow 0.3s;
        }
        .social-icons img:hover {
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);
        }
      </style>
      <div class="social-icons">
        <a href="https://github.com/Ahmad-Raza810" target="_blank">
            <img src="https://img.icons8.com/color/96/github.png" alt="GitHub">
        </a>
        <a href="https://www.linkedin.com/in/ahmad-raza-09062a323/" target="_blank">
            <img src="https://img.icons8.com/color/96/linkedin-circled.png" alt="LinkedIn">
        </a>
        <a href="https://www.instagram.com/ahhmad____77/" target="_blank">
            <img src="https://img.icons8.com/color/96/instagram-new.png" alt="Instagram">
        </a>
      </div>
      """
      st.markdown(social_media_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
