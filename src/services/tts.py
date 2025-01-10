import os
import tempfile
import streamlit as st
from openai import OpenAI

def generate_tts_audio(text, DEBUG=False):
    try:
        if DEBUG:
            print(f"\n=== Debug: TTS Generation ===")
            print(f"Input text: {text}")
            print(f"OpenAI API Key available: {'OPENAI_API_KEY' in os.environ}")
            
        client = OpenAI()
        
        if DEBUG:
            print("OpenAI client created")
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            if DEBUG:
                print(f"Created temp file: {temp_file.name}")
                
            try:
                response = client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="alloy",
                    input=text
                )
                if DEBUG:
                    print("TTS API call successful")
                    
                with response as r:
                    for chunk in r.iter_bytes():
                        temp_file.write(chunk)
                        
                if DEBUG:
                    print(f"Audio written to file: {temp_file.name}")
                    print(f"File size: {os.path.getsize(temp_file.name)} bytes")
                    
                return temp_file.name
                
            except Exception as api_error:
                if DEBUG:
                    print(f"API Error details: {str(api_error)}")
                    print(f"API Error type: {type(api_error)}")
                raise api_error
                
    except Exception as e:
        error_msg = f"Error generating audio: {str(e)}\nError type: {type(e)}"
        if DEBUG:
            print(f"Error in TTS generation: {error_msg}")
        st.error(error_msg)
        return None
