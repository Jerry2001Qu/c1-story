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

@st.cache_resource
def call_gpt(text):
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": text},
        ],
        n=1
    )
    return response.choices[0].message.content

def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Channel 1 Demo")

    if "data" not in st.session_state:
        st.session_state.data = pd.read_csv("cleaned.csv").drop(columns=["Unnamed: 0"])

    with st.container(border=True):
        story_id = st.selectbox(
            "Select Story",
            st.session_state.data.index,
            format_func=lambda x: st.session_state.data.Slug[x],
            key="story_selector"
        )
        selected_story = st.session_state.data.loc[story_id, "Main Content"]
        with st.expander(st.session_state.data.Title[story_id]):
            st.write(selected_story)

    submitted = st.button("Submit")

    # Initialize prompts if not in session state
    if "prompts" not in st.session_state:
        st.session_state.prompts = ["", "", ""]

    # Three columns for prompts
    cols = st.columns(3)

    for i, col in enumerate(cols):
        with col:
            st.session_state.prompts[i] = st.text_area(f"Prompt {i+1}", value=st.session_state.prompts[i], key=f"prompt_{i}")

    prompt_template = PromptTemplate.from_template(
"""{story}

\###

{instructions}""")

    if submitted:
        for i, prompt in enumerate(st.session_state.prompts):
            if not prompt:
                continue
            st.session_state.responses[i] = call_gpt(prompt_template.format(story=selected_story, instructions=prompt))

    # Display responses below each column
    for i, (col, response) in enumerate(zip(cols, st.session_state.responses.values())):
        with col:
            st.write(response)
        
    

if __name__ == "__main__":
    run()
