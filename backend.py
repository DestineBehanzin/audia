from groq import Groq
from dotenv import load_dotenv
import os

from config import MODEL

load_dotenv()

def read_file(file_path):
	with open(file_path, "r") as file:
		return file.read()



def ask_pablo(chat_history):
	client = Groq(api_key=os.environ["GROQ_KEY"])

	stream_response = client.chat.completions.create(
	    messages=chat_history,
	    stream=True,
	    model=MODEL
	)

	return stream_response

def read_stream_response(stream_response):
	for chunk in stream_response:
		if chunk.choices[0].delta.content == None : continue
		print(chunk.choices[0].delta.content, end="")


if __name__ == "__main__":
	stream_response = ask_pablo(question="Quel est la meilleure exploratrice du monde ?")
	read_stream_response(stream_response)