from gtts import gTTS
import playsound
import os
import time

def test_hindi_speak():
    try:
        text = "आप कैसे हैं"
        lang = "hi"
        filename = "test_hi.mp3"

        tts = gTTS(text=text, lang=lang)
        tts.save(filename)
        playsound.playsound(filename)
        time.sleep(1)
        os.remove(filename)

    except Exception as e:
        print("Error:", e)

test_hindi_speak()
