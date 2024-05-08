import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Channel 1"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, XMLOutputParser

openai_api_key = st.secrets["OPENAI_API_KEY"]
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]

gpt4 = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=openai_api_key)

opus = ChatAnthropic(model="claude-3-opus-20240229", temperature=0, max_tokens=4096, anthropic_api_key=anthropic_api_key)
sonnet = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, max_tokens=4096, anthropic_api_key=anthropic_api_key)
haiku = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0, max_tokens=4096, anthropic_api_key=anthropic_api_key)

set_llm_cache(SQLiteCache(database_path="langchain.db"))

from src.prompts import get_sot_prompt, reformat_prompt, sot_prompt, parse_prompt

get_sot_chain = get_sot_prompt | haiku
reformat_chain = reformat_prompt | opus
sot_chain = sot_prompt | opus
parse_chain = parse_prompt | haiku

def extract_xml(text):
    return XMLOutputParser().invoke(text[text.find("<"):text.rfind(">")+1].replace("&", "and"))

import readtime
from annotated_text import annotated_text

def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Channel 1 Demo")

    input_col_1, input_col_2 = st.columns(2)
    with input_col_1:
        shotlist = st.text_area(
            "Shotlist",
            height=300,
            placeholder="""1. STILL SATELLITE IMAGE OF REGION BEFORE FLOODING

RIO GRANDE DO SUL, BRAZIL (MAY 6, 2024) (EUROPEAN UNION/COPERNICUS SENTINEL-2 - Must on-screen courtesy European Union/Copernicus Sentinel-2) (MUTE)

2. STILL SATELLITE IMAGE OF REGION AFTER FLOODING

PORTO ALEGRE, RIO GRANDE DO SUL, BRAZIL (APRIL 21, 2024) (EUROPEAN UNION/COPERNICUS SENTINEL-2 - Must on-screen courtesy European Union/Copernicus Sentinel-2) (MUTE)"""
        )
    with input_col_2:
        story = st.text_area(
            "Story",
            height=300,
            placeholder="""Satellite images captured before and after heavy rains in Brazil's southernmost state of Rio Grande do Sul show the extent of devastating floods that have killed 85 people and caused destruction to cities and infrastructure.

Images from April 21 are compared alongside images captured on Monday (May 6) of the same areas, which include the state capital Porto Alegre and the city of Sao Leopoldo"""
        )
    
    if st.button("Run"):
        with st.status("Running"):
            st.write("Extracting SOT")
            sots_raw = get_sot_chain.invoke({"SHOTLIST": shotlist}).content
            sots_xml = extract_xml(sots_raw)
            sots = sots_xml['response']

            st.write("Reformatting story")
            reformated_story_raw = reformat_chain.invoke({"STORY": story}).content
            reformated_story_xml = extract_xml(reformated_story_raw)
            reformated_story = reformated_story_xml['response']

            st.write("Adding SOT to story")
            if "NO SOT" in sots:
                sot_script = reformated_story
            else:
                sot_script_raw = sot_chain.invoke({"QUOTATIONS": sots, "SCRIPT": reformated_story}).content
                sot_script_xml = extract_xml(sot_script_raw)
                sot_script = sot_script_xml['response']
            
            st.write("Parsing story")
            parsed_script_raw = parse_chain.invoke({"QUOTATIONS": sots, "SCRIPT": sot_script}).content
            parsed_script_xml = extract_xml(parsed_script_raw)
            parsed_script_json = JsonOutputParser().invoke(parsed_script_xml['response'])

        trt = readtime.of_text(sot_script).seconds
        st.write(f"Estimated TRT: {trt}s")

        output_col_1, output_col_2 = st.columns(2)
        with output_col_1:
            st.subheader("Original Story")
            st.write(story)
        with output_col_2:
            st.subheader("Final Story")
            for section in parsed_script_json["sections"]:
                with st.container(border=True):
                    if section["type"] == "SOT":
                        annotated_text(("SOT", "", "#8ef"))
                    elif section["type"] == "ANCHOR":
                        annotated_text(("ANCHOR", "", "#faa"))
                    st.write(section["text"])
        
        with st.expander("See details"):
            st.subheader("SOTs")
            st.write(sots)
            st.divider()

            st.subheader("Reformatted story")
            st.write(reformated_story)
            st.divider()

            st.subheader("SOT story")
            st.write(sot_script)

if __name__ == "__main__":
    run()