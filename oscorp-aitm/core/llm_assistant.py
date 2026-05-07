import os
import streamlit as st
from groq import Groq

LANGUAGES = ["English", "Hindi", "Kannada", "Marathi"]

class LLMAssistant:
    def __init__(self):
        self.client = None
        self.is_online = False
        
        # Look for the API key safely, even if secrets.toml is missing.
        api_key = None
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")

        if api_key:
            try:
                self.client = Groq(api_key=api_key)
                self.is_online = True
            except Exception as e:
                print(f"Groq Init Error: {e}")

    def generate_sentences(self, words: list, language: str) -> list:
        if not words:
            return []
            
        words_str = ', '.join(words)
        
        # If offline, return a fake response to prove the UI works
        if not self.is_online:
            return [
                f"[OFFLINE] No API key found.",
                f"I saw you say: {words_str}",
                "Please check secrets.toml"
            ]

        prompt = (
            f"Based on the following word or words: '{words_str}', generate 3 very common, "
            f"daily-use conversational sentences in {language}. "
            f"Provide ONLY the sentences, separated by a newline (\\n)."
            f"Do NOT include numbers or bullet points."
        )

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.7,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Split by lines and clean up
            sentences = [s.strip() for s in content.split('\n') if s.strip()]
            
            clean_sentences = []
            for s in sentences:
                if s[0].isdigit() and s[1] in ['.', ')']:
                    s = s[2:].strip()
                elif s.startswith('-') or s.startswith('*'):
                    s = s[1:].strip()
                clean_sentences.append(s)
            
            # If Groq returned a weird format that got cleaned out, force a fallback
            if not clean_sentences:
                return [f"Groq returned a weird format: {content}"]
                
            return clean_sentences[:3]
            
        except Exception as e:
            # THIS IS CRITICAL: Send the error text back to the UI!
            error_msg = str(e)
            print(f"Groq generation error: {error_msg}")
            return [f"API ERROR: {error_msg[:100]}..."]