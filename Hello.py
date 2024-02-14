# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI

gpt4 = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview")

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_data
def call_gpt(text):
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": text},
        ],
        n=3
    )
    return response.choices

def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Channel 1 Demo")

    if "data" not in st.session_state:
        st.session_state.data = pd.read_csv("cleaned.csv").drop(columns=["Unnamed: 0"])
    if "story_index" not in st.session_state:
        st.session_state.story_index = 0

    st.session_state.story_index = st.sidebar.selectbox(
        "Prefill Stories",
        st.session_state.data.index,
        format_func=lambda x: st.session_state.data.loc[x].Slug
    )

    story = st.text_area(
        "Story",
        value=st.session_state.data["Main Content"][st.session_state.story_index],
        height=500
    )

    instructions = st.text_area(
        "Instructions",
        value="Given this info, write a news script that's about 300 words. Include quotes where possible with SOT: at the beginning of the paragraph. Never make up information.",
    )

    prompt_template = PromptTemplate.from_template(
"""{story}

\###

{instructions}""")

    with st.expander("AI Input", expanded=False):
        st.write(prompt_template.format(story=story, instructions=instructions))

    submitted = st.button("Submit")

    if submitted:
        st.session_state.responses = call_gpt(prompt_template.format(story=story, instructions=instructions))
        cols = st.columns(3)

        for i, (col, response) in enumerate(zip(cols, st.session_state.responses)):
            with col:
                st.subheader(f"{i+1}")
                st.write(response.message.content)
    
    if "responses" in st.session_state:
        cols = st.columns(3)

        for i, (col, response) in enumerate(zip(cols, st.session_state.responses)):
            with col:
                st.subheader(f"Script {i}")
                st.write(response.message.content)
        
    

if __name__ == "__main__":
    run()
