from langchain.prompts import PromptTemplate

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

Please give me a numbered list of shots in the shotlist that contain quotations. Don't copy from the examples. Copy from
the shotlist exactly and put it in <response></response> tags. If there are no quotes respond with <response>NO SOT</response>""")

reformat_prompt = PromptTemplate.from_template(
"""I'm producing a television news segment.  I'd like to reformat a news story I wrote so it 
can be spoken by an on camera news anchor. Please don't change any of the facts of 
my original story text at all, and please don't change the wording at all.  Just reformat it 
so a TV news anchor could easily read it aloud.  Any dates need to be formatted so 
they can be read aloud. For example, May 1st instead of May 1.

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

Using only the list of clips that contain quotations, insert those quotations into the 
news script where it feels appropriate to the story, and in a way that makes sense to 
the overall flow of the story. But otherwise don't change the wording of the news script 
at all. The quotations should be spread out throughout the story. You may start the 
story with a quotation if it makes sense to do so, but don't end the story with a 
quotation. Keep the quotations short, you may cut out a portion to keep it short.
Insert quotations exactly with descriptions. Don't add transitions. Think through
your answer before responding, including if it makes sense to start with a quote. Put the text in <response></response> tags."""
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

Each paragraph should become a section. Some paragraphs are quotes. These should be matched with quotes from the shotlist & add a shot_id.
Put the output in <response></response> tags"""
)

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