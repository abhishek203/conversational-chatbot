import speech_recognition as sr


r = sr.Recognizer()
m = sr.Microphone()

print("what do you want to say..")

with m as source:

    audio = r.listen(source,phrase_time_limit=10)

    try:
        text = r.recognize_google(audio)
        print(text)
    except:
        print("Error")