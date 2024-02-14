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
import os

gpt4 = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model="gpt-4-turbo-preview")

def run():
    st.set_page_config(
        page_title="Channel 1",
        page_icon="👋",
    )

    st.write("# Channel 1 Demo 👋")

    df = pd.read_csv("cleaned.csv").drop(columns=["Unnamed: 0"])
    story_index = 0

    with st.expander("Story Selector"):
      story_index = st.selectbox(
          "Story Selector",
          df.index,
          format_func=lambda x: df.loc[x].Title
      )

      df.loc[story_index]

    story = st.text_area(
        "Story",
        value=df["Main Content"][story_index],
        height=500
    )

    prompt = st.text_area(
        "Prompt",
        value="Given this info, write a news story that's about 300 words. Include quotes where possible marked as SOT:",
    )

    llm_input = f"{story}\n\n\####\n\n{prompt}"

    with st.expander("AI Input", expanded=False):
        st.write(llm_input)

    # if st.button("Submit"):
    #     gpt4
    

if __name__ == "__main__":
    run()
