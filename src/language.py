# Language
# STREAMLIT
from src.prompts import run_chain, language_to_iso_chain
# /STREAMLIT

import pycountry
from dataclasses import dataclass

@dataclass
class Language:
    alpha_2: str
    name: str

    @classmethod
    def from_str(cls, language_str):
        """Consolidates language from either full name, ISO-639-1, or ISO-639-3"""
        if len(language_str) == 2:
            language = pycountry.languages.get(alpha_2=language_str)
        elif len(language_str) == 3:
            language = pycountry.languages.get(alpha_3=language_str)
        else:
            language = pycountry.languages.get(name=language_str)

        if not language:
            print(f"No language match for: {language_str}, attempting LLM matching.")
            language_code_2 = run_chain(language_to_iso_chain, {"LANGUAGE_NAME": language_str})
            if language_code_2 == "Unknown":
                print(f"Language could not be matched: {language_str}")
                return cls("Unknown", "Unknown")
            else:
                language = pycountry.languages.get(alpha_2=language_code_2)
        return cls(language.alpha_2, language.name)