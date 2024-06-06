# Prompts

# STREAMLIT
from src.constants import LANGCHAIN_API_KEY, ANTHROPIC_API_KEY
# /STREAMLIT

import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Channel 1"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, XMLOutputParser
from langchain.prompts import PromptTemplate

opus = ChatAnthropic(model="claude-3-opus-20240229", temperature=0, max_tokens=4096, anthropic_api_key=ANTHROPIC_API_KEY)
sonnet = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, max_tokens=4096, anthropic_api_key=ANTHROPIC_API_KEY)
haiku = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0, max_tokens=4096, anthropic_api_key=ANTHROPIC_API_KEY)

cache = SQLiteCache(database_path="langchain.db")
set_llm_cache(cache)

# STREAMLIT
section_summary_prompt = PromptTemplate.from_template(
"""Write a summary for each section in my script.

<example>
{{
  "sections": [
    {{
      "id": 1,
      "summary": "Eurovision culminates in Sweden this weekend."
    }},
    {{
      "id": 2,
      "summary": "100K visitors from 89 nations in host Malmo."
    }},
    {{
      "id": 3,
      "summary": "Eurovision resisted calls to exclude Israel."
    }},
    {{
      "id": 4,
      "summary": "Israel's Eden Golan is bookies' favourite."
    }},
  ]
}}
</example>

Here is the script:
<script>
{SCRIPT}
</script>

Write a summary for each section. The first section summary should summarize the key story. Keep summaries short, less than 40 characters.
Put the output in <response></response> tags."""
)
# /STREAMLIT

facts_prompt = PromptTemplate.from_template(
"""Give me a complete list of facts from this story.

Here is the script:
<script>
{SCRIPT}
</script>

Here is the shotlist:
<shotlist>
{SHOTLIST}
</shotlist>

Return a numbered list with the most important facts at the start. Try to capture every fact in the script & shotlist. Put your response in <response></response> tags."""
)

get_sot_prompt = PromptTemplate.from_template(
"""I'm trying to extract shots with quotations from a shotlist.

<example>
1. PROTESTERS SHOUTING (English): "THEY'RE COMING FOR US, HOLD THE LINE"
3. PROTESTER SHOUTING (English) "FREE PALESTINE" WHILE BEING TAKEN AWAY BY POLICE
12. PROTESTER SAYING (English) "I AM A STUDENT HERE, I AM AN ENGLISH MAJOR. PLEASE DON'T FAIL US, DON'T FAIL US, YOU ARE ALREADY FAILING US" WHILE BEING TAKEN AWAY BY POLICE
</example>

<shotlist>
{SHOTLIST}
</shotlist>

Please give me a numbered list of shots in the shotlist that contain quotations. Don't copy from the examples. 
Include the entire section from the shotlist including parts like :(SOUNDBITE)(English) LOREM IPSUM, SAYING:".
Copy from the shotlist exactly and put it in <response></response> tags. If there are no quotes respond with <response>NO SOT</response>.""")

parse_sot_prompt = PromptTemplate.from_template(
"""You will be given quotes that will be aired on the news. Your task is to extract key information from the quote.

Here are the quotes:
<quotes>
{QUOTES}
</quotes>

Please extract the following information from the quote:
- The ID
- The verbatim text of what was said
- The name of the person who said it, title case (if not available, put "Unknown")
- The title of the person who said it, title case (e.g. "Protester", "Former President", etc. If no title is available, just put "Unknown")
- The language the quote was spoken in (English if unspecified)

Provide your extractions in JSON format. Use the ID as a key. For each object use the keys "quote", "name", "title", and "language" for the respective pieces of information.
Ensure quotes are properly escaped or replaced with single quotes to ensure JSON is valid.

<example>
{{
  "1": {{
      "quote": "...",
      "name": "...",
      "title": "CEO and President",
      "language": "english",
  }},
  "4": {{
      "quote": "...",
      "name": "...",
      "title": "...",
      "language": "chinese",
  }},
}}
</example>

Provide the extractions inside <response></response> tags.""")

reformat_prompt = PromptTemplate.from_template(
"""I'm producing a television news segment.  I'd like to reformat a news story I wrote so it
can be spoken by an on camera news anchor. Please don't change any of the facts of
my original story text at all, and please don't change the wording at all.  Just reformat it
so a TV news anchor could easily read it aloud.

- Any dates need to be formatted so they can be read aloud. For example, May 1st instead of May 1.
- Remove any parentheses and the contained text.
- Remove any lone hyphens (Em Dashes). Don't remove hyphens that connect two words.

Here is the story:
<story>
{STORY}
</story>

Reformat the story and put it in <response></response> tags"""
)

sot_prompt = PromptTemplate.from_template(
"""I'm producing a television news segment. I'd like to insert clips of quotations into a news script.

<example>
POLICE GATHERING AT THE ENTRANCE OF A CONCERT VENUE,
FANS YELLING (English): "THEY'RE SHUTTING IT DOWN, STAY TOGETHER"

Scores of police officers assembled at the gates of the Madison Square Garden late Saturday night, intent on halting a surprise concert that had spiraled out of control due to overwhelming attendance.

Live streaming on social media depicted officers in uniform clearing the area and confronting fans who had scaled fences to gain entry.

FAN SHOUTING (English) "MUSIC IS OUR RIGHT" AS POLICE LEAD THEM AWAY

The air was thick with the sound of sirens as the crowd grew increasingly agitated, resisting the police’s efforts to disperse the impromptu gathering.

Fans, some equipped with concert posters and glow sticks, tried to prevent the police from entering the venue by forming human chains, all while chanting, "let us sing" and shining flashlights towards the officers.

FAN EXCLAIMING (English) "I’VE WAITED MONTHS FOR THIS, YOU CAN’T JUST TAKE IT AWAY" WHILE BEING ESCORTED OUT BY POLICE

A few individuals near the back of the crowd chose not to resist, and footage showed them walking towards the exits with their hands visibly empty and raised.

The unexpected police intervention at Madison Square Garden became a pivotal moment, showcasing the challenges faced in managing unexpectedly large crowds at major events.
</example>

Here are the clips of quotations:
<quotations>
{QUOTATIONS}
</quotations>

Here is the script:
<script>
{SCRIPT}
</script>

Please carefully read through the list of quotation clips and the news script.

In a <scratchpad>, think through where it would make the most sense to insert each quotation into the news script in a way that fits the overall story and flow. Remember to:

- Keep the quotations short, cutting out portions if needed to keep them concise. No more than 1 or 2 sentences.
- Spread the quotations throughout the story
- You may start the story with an English quotation if it makes sense to do so. But don't with a non-English quotation
- Do not end the story with a quotation
- Insert the portions of quotations including their descriptions
- Do not add any transitions between the quotations and the rest of the script
- Do not change the wording of the news script except to insert the quotations
- Place quotations in separate paragraphs. You may split paragraphs in the original script.

After planning it out, provide your final response with the quotations inserted into the appropriate parts of the news script inside <response></response> tags."""
)

parse_prompt = PromptTemplate.from_template(
"""Parse my script into a JSON object.

<example>
{{
  "sections": [
    {{
      "id": 1,
      "text": "PROTESTERS GATHERED IN THE STREET, SHOUTING (English): \"NO JUSTICE, NO PEACE!\""
      "type": "SOT",
      "shot_id": 2,
      "quote": "NO JUSTICE, NO PEACE!",
    }},
    {{
      "id": 2,
      "text": "Hundreds of climate activists assembled outside the State Capitol early on Thursday, May second, to demand immediate government action against climate change."
      "type": "ANCHOR",
    }},
    {{
      "id": 3,
      "text": "Live TV footage showed protesters chanting slogans and carrying banners demanding policy changes to tackle environmental issues. Police in tactical gear maintained a tight perimeter around the Capitol building."
      "type": "ANCHOR",
    }},
    {{
      "id": 4,
      "text": "PROTESTER SHOUTING (English): \"CLIMATE JUSTICE NOW!\" WHILE WAVING A BANNER",
      "type": "SOT",
      "shot_id": 8,
      "quote": "CLIMATE JUSTICE NOW!"
    }},
  ]
}}
</example>

Here are the shots with quotes:
<quotations>
{QUOTATIONS}
</quotations>

Here is the script:
<script>
{SCRIPT}
</script>

 - Each paragraph should become a section
 - Some paragraphs are quotes. These should be matched with quotes from the shotlist & add a shot_id
 - There may not be any quotes
 - Only put speech in the quote field, no emotes like (laughs)
 - For quotes, always remove or replace parentheses/brackets and the contained text. For example this should be fully removed: (local time / 0200EST). Acronyms can be replaced such as: Korean Demilitarized Zone (DMZ) -> Korean Demilitarized Zone or DMZ.
 - Remove any lone hyphens (Em Dashes). Don't remove hyphens that connect two words.
 - Ensure quotes are properly escaped or replaced with single quotes to ensure JSON is valid
 - Ensure the order remains the same between all paragraphs (SOT & ANCHOR)
 - Don't move all the SOTs to the end

Put the output in <response></response> tags"""
)

logline_prompt = PromptTemplate.from_template(
"""You are a TV script writer. Here is a list of text sections:

<sections>
{SECTIONS}
</sections>

For each section, write a logline/summary sentence that is no more than 40 characters long. Capture the key point of the section concisely and densely! Only use acronyms that are commonly known. Don't too many acronyms (2 max).

Return your response as a JSON object with a list of sections. Each section object should have an "id" field containing the section number (e.g. 0, 2, etc.)
and a "logline" field containing the logline/summary you wrote for that section.

<example>
{{
  "sections": [
    {{
      "id": 1,
      "logline": "..."
    }},
    {{
      "id": 2,
      "logline": "..."
    }},
  ]
}}
</example>

Enclose your final JSON response inside <response></response> tags."""
)

headline_prompt = PromptTemplate.from_template(
"""Here is the script for a news story:

<script>
{SCRIPT}
</script>

Please read the story script carefully. Then, in a <brainstorming> tag, come up with 3-5 potential headlines for this story. The headlines should:
- Summarize the key points of the story
- Be concise (MUST BE between 35 and 48 characters), and content dense
- Be written in an attention-grabbing style that would make readers want to click on the story
- Is informative over all else
- Be in title case

After brainstorming potential headlines, select the one you think is best (count each character count) and output it in <response></response> tag."""
)

parse_broll_prompt = PromptTemplate.from_template(
"""Parse this list of broll placements into JSON.

<example>
{{
  "sections": [
    {{
      "id": 1,
      "brolls": [
        {{
            "id": "Anchor",
            "start": 0.00,
            "end": 4.00
        }},
        {{
            "id": "004",
            "start": 4.00,
            "end": 9.21,
        }},
      ]
    }},
    {{
      "id": 3,
      "brolls": [
        {{
            "id": "005",
            "start": 0.00,
            "end": 6.00,
        }},
        {{
            "id": "011",
            "start": 6.00,
            "end": 10.16,
        }},
      ]
    }},
  ]
}}
</example>

Here are the section start and end times:
<sections>
{SECTIONS}
</sections>

Here is the list of broll placements:
<broll_placements>
{BROLL_PLACEMENTS}
</broll_placements>

Parse the <broll_placements></broll_placements> into each <section></section>. Start each section at 0. Put your response in <response></response> tags."""
)

match_sot_prompt = PromptTemplate.from_template(
"""Your task is to find the substring in one language that best matches the meaning of a string in English.

Here is the English string to match:
<english>
{ENGLISH_STRING}
</english>

And here is the text in another language to find the matching substring in:
<other_language>
{OTHER_LANGUAGE_STRING}
</other_language>

Your resulting substring should be contiguous & be taken exactly from the other language. It should be in that other language.
If this doesn't match any language name, return Unknown. Put your response in <response></response> tags."""
)

match_hard_sot_prompt = PromptTemplate.from_template(
"""Your task is to fuzzy find a quote within a larger transcript. The transcript is automatically generated while the quote is 
a human written. Therefore, they will not match word for word.

Here is the quote to match:
<quote>
{QUOTE}
</quote>

And here is the larger transcript to find the matching substring in:
<transcript>
{TRANSCRIPT}
</transcript>

Your resulting fuzzy match should be contiguous & be taken exactly from the transcript. Try to obtain as much of the quote as 
possible. If there is no fuzzy match, return Unknown. Put your final response in <response></response> tags."""
)

language_to_iso_prompt = PromptTemplate.from_template(
"""Your task is to convert a given language name to its corresponding ISO 639-1 language code.

The language name to convert is:
<language_name>
{LANGUAGE_NAME}
</language_name>

If this doesn't match any language name, return Unknown. Put your response in <response></response> tags."""
)

match_clip_to_sots_prompt = PromptTemplate.from_template(
"""You will be provided with two pieces of information:

<sots>
{SOTS}
</sots>

<clips_with_transcripts>
{CLIPS_WITH_TRANSCRIPTS}
</clips_with_transcripts>

Your task is to match each clip from the clips_with_transcripts to the most relevant sot from the sots. Not every clip will necessarily have a matching sot.

Return a JSON in a format like:
<example>
{{
   "matches": [
      {{
         "clip_id": <clip_id>,
         "sot_id": <sot_id>,
         "shotlist_description": <sot_description>
      }},
      ...
   ]
}}
</example>

The transcript from a clip should match the quote from a sot to match. If a clip doesn't match any sot, set sot_id to None.
If the clip matches an sot, copy it's description from the sots list, otherwise None.
Put your response in <response></response> tags."""
)

get_sot_chain = get_sot_prompt | opus
facts_chain = facts_prompt | opus
parse_sot_chain = parse_sot_prompt | opus
reformat_chain = reformat_prompt | opus
sot_chain = sot_prompt | opus
parse_chain = parse_prompt | opus
logline_chain = logline_prompt | opus
headline_chain = headline_prompt | opus
parse_broll_chain = parse_broll_prompt | opus
match_sot_chain = match_sot_prompt | opus
match_hard_sot_chain = match_hard_sot_prompt | opus
language_to_iso_chain = language_to_iso_prompt | opus
match_clip_to_sots_chain = match_clip_to_sots_prompt | opus

from sqlalchemy.exc import OperationalError
import time
from anthropic import APIError

def extract_xml(text):
    return XMLOutputParser().invoke(text[text.find("<response>"):text.rfind("</response>")+11].replace("&", "and"))

def run_chain(chain, params, max_retries=3, retry_delay=5):
    """Runs the LangChain chain with retry logic."""
    retries = 0
    while retries <= max_retries:
        try:
            response_raw = chain.invoke(params).content
            response_xml = extract_xml(response_raw)
            return response_xml['response'].strip()
        except OperationalError:
            cache = SQLiteCache(database_path="langchain.db")
            set_llm_cache(cache)
            response_raw = chain.invoke(params).content
            response_xml = extract_xml(response_raw)
            return response_xml['response'].strip()
        except APIError as e: 
            if e.status_code == 529 and retries < max_retries: 
                print(f"Server overloaded, retrying in {retry_delay} seconds...")
                retries += 1
                time.sleep(retry_delay)
            elif retries < max_retries:
                retries += 1
                time.sleep(retry_delay)
            else:
                raise e  # Re-raise if retries exceeded

def run_chain_json(chain, params):
    response = run_chain(chain, params)
    return JsonOutputParser().invoke(response)