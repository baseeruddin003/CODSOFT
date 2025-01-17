from datetime import datetime

def chatbot():
    print("Chatbot: Hi there! I'm your friendly assistant. Ask me anything or type 'exit' to leave the chat.")

    while True:
        user_input = input("You: ").strip().lower()

        if user_input in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! Take care!")
            break

        if "hello" in user_input or "hi" in user_input or "hey" in user_input:
            print("Chatbot: Hello! How can I help you today?")
        elif "how are you" in user_input:
            print("Chatbot: I'm just a chatbot, but I'm doing great! How about you?")
        elif "what's your name" in user_input or "your name" in user_input:
            print("Chatbot: I'm your helpful assistant. You can call me ChatBot!")
        elif "time" in user_input or "date" in user_input:
            now = datetime.now()
            print(f"Chatbot: Right now, it's {now.strftime('%A, %B %d, %Y %I:%M %p')}.")
        elif "thank you" in user_input or "thanks" in user_input:
            print("Chatbot: You're welcome! Always happy to help.")
        elif "help" in user_input:
            print("Chatbot: I can chat with you, tell jokes, give the current time and date, and answer basic questions.")

        elif "joke" in user_input:
            print("Chatbot: Why did the scarecrow win an award? Because he was outstanding in his field!")
        elif "weather" in user_input:
            print("Chatbot: I can't fetch live weather updates yet")
        elif "news" in user_input:
            print("Chatbot: I'm not integrated with a news service yet.")
        elif "hobby" in user_input or "interests" in user_input:
            print("Chatbot: I love listening to music. What are your hobbies?")
        elif "tell me about yourself" in user_input:
            print("Chatbot: I'm a chatbot built to make your life easier by answering questions and keeping you company. What would you like to know?")

        elif "movie" in user_input or "recommend a movie" in user_input:
            print("Chatbot: I like 'Interstellar'. What kind of movies do you enjoy?")
        elif "fun fact" in user_input:
            print("Chatbot: Here's a fun fact: Did you know octopuses have three hearts?")
        elif "food" in user_input or "what to eat" in user_input:
            print("Chatbot: How about trying something new? A good pizza sounds great!")
        
        elif "motivation" in user_input or "inspiration" in user_input:
            print("Chatbot: Keep going! Every step forward is a step toward achieving your goals. You've got this!")
        elif "who am i" in user_input or "identity" in user_input:
            print("Chatbot: You are a unique and amazing individual with endless potential!")
        elif "favorite color" in user_input:
            print("Chatbot: My fav colour is blue. What's your favorite color?")
        else:
            print("Chatbot: Hmm, I didn't catch that. Could you say it differently?")



if __name__ == "__main__":
    chatbot()
