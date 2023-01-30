from gtts import gTTS

def texttospeech(mytext):
    # Import the required module for text to speech conversion 

  
    # This module is imported so that we can play the converted audio 
    import os 
   
    # Language in which you want to convert 
    language = 'en'
  
    # Passing the text and language to the engine, here we have marked slow=False. Which tells  the module that the converted audio should have a high speed 
    output = gTTS(text=mytext, lang=language, slow=False) 
  
    # Saving the converted audio in a mp3 file named welcome  
    output.save("speech.mp3") 
  
    # Playing the converted file 
    os.system("speech.mp3")

