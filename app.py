import streamlit as st
import nltk
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

model_name = "gpt2"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, framework="pt") 
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

def healthcare_chatbot(user_input):
    user_input = user_input.lower()

    if any(word in user_input for word in ["symptom", "sick", "flu", "fever", "pain"]):
        return (
            "**It seems like you're experiencing symptoms. Here’s what you can do:**\n"
            "- Get enough rest and stay hydrated.\n"
            "- Monitor your symptoms for any changes.\n"
            "- If symptoms persist or worsen, consult a healthcare professional.\n\n"
            "Would you like recommendations for home remedies or nearby clinics? (Yes/No)"
        )
    elif any(word in user_input for word in ["appointment", "schedule", "doctor visit"]):
        return (
            "**You can schedule an appointment with a doctor.**\n"
            "- Check for available time slots.\n"
            "- Select a suitable date and time.\n"
            "- Confirm your booking.\n\n"
            "Would you like assistance in finding a nearby doctor? (Yes/No)"
        )
    elif any(word in user_input for word in ["medication", "medicine", "drug", "prescription"]):
        return (
            "**It's important to take prescribed medicines correctly.**\n"
            "- Follow the dosage instructions carefully.\n"
            "- Store medicines in a cool, dry place.\n"
            "- Never share prescription medications.\n\n"
            "Would you like to set a reminder for your medication? (Yes/No)"
        )
    elif any(word in user_input for word in ["emergency", "urgent", "critical", "ER"]):
        return (
            "**If this is a medical emergency, call emergency services immediately.**\n"
            "- If experiencing severe symptoms like chest pain or difficulty breathing, seek immediate medical attention.\n\n"
            "Do you need a list of emergency contacts or nearby hospitals? (Yes/No)"
        )
    else:
        try:
            response = chatbot(user_input, max_length=150, num_return_sequences=1)
            return (f"{response[0]['generated_text']}\n\nWould you like more details on this topic? (Yes/No)")
        except Exception as e:
            return f"Error generating response: {str(e)}"

def main():
    st.set_page_config(page_title="Healthcare Assistant Chatbot", page_icon="💬", layout="wide")
    
    st.sidebar.title("🔍 Chatbot Features")
    st.sidebar.info("Ask me about symptoms, medications, or book an appointment.")
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Quick Health Tips:")
    st.sidebar.markdown("✔️ Stay hydrated and eat a balanced diet.")
    st.sidebar.markdown("✔️ Exercise regularly for good health.")
    st.sidebar.markdown("✔️ In case of emergency, seek medical help immediately.")
    
    st.title("💬 Healthcare Assistant Chatbot")
    st.write("Welcome! I am here to assist you with healthcare-related queries.")

    user_input = st.text_input("💡 How can I assist you today?", "")

    if st.button("Send Message "): 
        if user_input.strip():
            st.markdown(f"**👤 You:** {user_input}")
            with st.spinner("⏳ Processing your query, please wait..."):
                response = healthcare_chatbot(user_input)
                st.markdown(f"**Assistant:** {response}")
        else:
            st.warning(" Please enter a query.")

if __name__ == "__main__":
    main()
