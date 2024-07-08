# Prompts

# STREAMLIT
from src.constants import LANGCHAIN_API_KEY, ANTHROPIC_API_KEY
import streamlit as st
# /STREAMLIT

import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Channel 1"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser, XMLOutputParser
from langchain.prompts import PromptTemplate

opus = ChatAnthropic(model="claude-3-opus-20240229", temperature=0, max_tokens=4096, anthropic_api_key=ANTHROPIC_API_KEY)
sonnet = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, max_tokens=4096, anthropic_api_key=ANTHROPIC_API_KEY)
haiku = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0, max_tokens=4096, anthropic_api_key=ANTHROPIC_API_KEY)

sonnet35 = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0, max_tokens=4096, anthropic_api_key=ANTHROPIC_API_KEY)

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

spell_check_prompt = PromptTemplate.from_template(
"""Fix any spelling mistakes in the following text.

<input>
{INPUT}
</input>

- Only fix obvious spelling mistakes that you are sure of.
- If you are somewhat uncertain about a mistake, do not change it.
- Do not fix grammar. You may only fix spelling mistakes one word at a time.
- Do not change punctuation/spacing.
- Do not localize spelling. For example, do not change Labour to Labor.
    - Labour -> Labour
- Do not change names. Assume they are spelled correctly.
- Otherwise, output the text just as it came in without any other changes.

Put your response in <response> tags."""
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

Please give me a numbered list of shots in the shotlist that contain quotations. The numbers should be from the original shotlist. Don't copy from the examples.
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
Ensure double quotes in strings are replaced with single quotes to ensure JSON is valid.

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
"""I'm producing a television news segment. I'd like to reformat a news story I wrote so it
can be spoken by an on camera news anchor. Please don't change any of the facts of
my original story text at all, and please don't change the wording unless specified.
Just reformat it so a TV news anchor could easily read it aloud.

The script may contain soundbites. They are in a format like:
(SOUNDBITE) (Urdu) LOCAL RESIDENT, MOHAMMAD IMRAN, SAYING:
"People in Karachi are facing extreme heat at the moment. Look..."

- DO NOT touch soundbites.
  - Do not remove any formatting around soundbites. Leave the "(SOUNDBITE) (Urdu) LOCAL RESIDENT, MOHAMMAD IMRAN, SAYING:"
  - Do not change any soundbites. Ignore the rest of the instructions for soundbites.
- Always remove or replace parentheses/brackets and the contained text.
  - For example this should be fully removed: (local time / 0200EST).
  - Acronyms can be replaced such as: Korean Demilitarized Zone (DMZ) -> Korean Demilitarized Zone or DMZ.
  - Ensure the wording flows when spoken aloud. 100,000 rupees ($358) -> One hundred thousand rupees, or three hundred fifty eight dollars.
- Convert abbreviations that sound better spoken in full.
  - John v. Doe -> John versus Doe
  - St. Louis -> Saint Louis
  - Jan -> January
  - Don't convert abbreviations that are more commonly spoken as abbreviations.
    - U.S. -> U.S.
    - NASA -> NASA
    - FBI -> FBI
    - Can't -> Can't
- Remove ALL lone hyphens (Em Dashes). Don't remove hyphens that connect two words.
- Convert ALL numbers to spoken word. Only add hyphens for numbers between 21 and 99.
  - 1,200 -> One thousand two hundred
  - 1,254 -> One thousand two hundred fifty-four
- ALL dates need to be formatted so they can be read aloud.
  - May 1 -> May 1st
- Days of week with dates within the current week should remove the date. Leaving the day of the week. If the date is not within the current week, leave the date.
  - Monday, May 1 -> Monday
  - on Tuesday (June 25) -> on Tuesday
- Reformat ages to be spoken aloud. Add hyphens.
  - John Doe, 23, said -> Twenty-three-year-old John Doe said
- ALWAYS Move "... said" to the beginning of the sentence if it is not already there. The individual/group should come before what was said.
  - ("We do not even have half of the people we should have, " Oleg, 49-year-old gunner said.) -> (Forty nine year old gunner, Oleg said, "We do not even have half of the people we should have.")
  - (The offensive has left Gaza in ruins, killing more than 37,600 people, according to Gaza's health ministry.) -> (According to Gaza's health ministry, the offensive has left Gaza in ruins, killing more than 37,600 people.)
  - (He will return to Australia after the hearing, according to a Wikileaks statement.) -> (According to a Wikileaks statement, he will return to Australia after the hearing.)
  - (Extreme temperatures throughout Asia over the past month were made worse most likely as a result of human-driven climate change, a team of international scientists have said.) -> (A team of international scientists have said, extreme temperatures throughout Asia over the past month were made worse most likely as a result of human-driven climate change.)
  - Don't combine quotes while doing this. Create a new paragraph if two quotes are adjacent.
- Fix any small grammatical and spelling errors that would make the story easier to read aloud.
  - She will be work in the US -> She will be working in the US
- Make any other changes that would aid in the story being read aloud, word for word.

Current Date:
<date>
{DATE}
</date>

Here is the story:
<story>
{STORY}
</story>

Reformat the story and put it in <response></response> tags
"""
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

In a <scratchpad>, think through where it would make the most sense to insert quotations into the news script in a way that fits the overall story and flow. Remember to:

- Keep the quotations short, extracting just a portion of the original quote if needed to keep them concise. No more than 1 or 2 sentences.
- If extracting a portion of a quotation, ensure the portion is CONTIGUOUS. Never cut a gap out of the portion.
- Only insert quotations that are relevant, add value, and are interesting to the viewer. If it's not interesting, don't insert it.
- Spread the quotations throughout the story
- Aim for 2-3 quotations inserted. You do not have to use all of the quotations.
- You may start the story with an English quotation if it makes sense to do so. But don't with a non-English quotation
- NEVER put a quotation at the end of the entire story. We must end with the script
- Insert the portions of quotations including their descriptions
- Do not add any transitions between the quotations and the rest of the script
- Do not change the wording of the news script except to insert the quotations
- Place quotations in separate paragraphs. You may split paragraphs in the original script.

After planning it out, provide your final response with the quotations inserted into the appropriate parts of the news script inside <response></response> tags."""
)

edit_prompt = PromptTemplate.from_template(
"""I'm producing a television news segment. You are an expert editor which will edit my news script.

Here is the original script:
<script>
{SCRIPT}
</script>

You will edit this script to maintain a proper 'flow', while removing unnecessary/irrelavent information.

You may only make simple edits, including:
- Reordering sections
- Removing sections
- Grammatical changes that don't change meaning
  - But like most frontline units, Vasil's team suffers from shortages of manpower. -> But most frontline units still suffer from shortages of manpower.

- Don't change the meaning of any sentences. Never state what someone else said as your own.
- The anchor should not give opinions. Opinions can be in quotes/statements from other individuals.
- Ensure your changes are factual & do not misrepresent the original script.
- Remember the current date. Past events should be in past tense.

The script may contain soundbites. They are in a format like:
(SOUNDBITE) (Urdu) LOCAL RESIDENT, MOHAMMAD IMRAN, SAYING:
"People in Karachi are facing extreme heat at the moment. Look..."

- Do not remove any formatting around soundbites. Leave the "(SOUNDBITE) (Urdu) LOCAL RESIDENT, MOHAMMAD IMRAN, SAYING:"
- You may reorder, remove, or cut down soundbites.
- If cutting down a soundbite, ensure the soundbite is CONTIGUOUS. Therefore, you may only cut from the start or end of a soundbite!
- Soundbites are better when spread throughout the story instead of clumped together.

<example>
<example_input>
Ukrainian gunners at the frontline in Donetsk region say that after months of ammunition shortages they finally have enough shells to fight with.

Forty-nine-year-old Vasil, commander of an artillery unit of the Thirty-Third Separate Mechanised Brigade said on Sunday, "There was a 'shell hunger', ammunition was rationed quite severely."

(SOUNDBITE) (UKRAINIAN)  VASIL, 46, CALL SIGN SILVER, COMMANDER OF THE UNIT OPERATING M-109 HOWITZER, SAYING: 
"There was a 'shell hunger', ammunuition was rationed quite severely. It had an impact on infantry, they (Russians) crept from all sides, it hurt the infantry men. Now, there is no more 'shell hunger' and we work well."

"Now, there is no more 'shell hunger' and we work well," he said as his fellow soldiers fired the one hundred fifty-five millimeter self-propelled howitzer M-109.

Demand for one hundred fifty-five millimeter artillery rounds has soared since Russia invaded Ukraine in February 2022. Allies' supplies for their own defense have been run down as they have rushed shells to Kyiv, which fires thousands of rounds per day.

Fresh influx of ammunition began arriving at units like Vasil's after a sixty-one billion dollar U.S. aid package was approved in April.

But like most frontline units, Vasil's team suffers from shortages of manpower.

(SOUNDBITE) (UKRAINIAN) OLEG, 49 GUNNER SAYING: 
"We are very few, there aren't enough people. We do not even have half of the people we should have."

Forty-nine-year-old gunner, Oleg said, "We do not even have half of the people we should have."

Outnumbered and exhausted after two and a half years in the trenches with little hope of being rotated any time soon, many are determined to keep going.

(SOUNDBITE) (UKRAINIAN)  VASIL, 46, CALL SIGN SILVER, COMMANDER OF THE UNIT OPERATING M-109 HOWITZER, SAYING: 
"All these talks with Korea and China, they will not help them, we will win, we shall overcome. It is our spirit, it is our Ukraine, we are defending it. We shall overcome, at any price but we will win."

"At any price but we will win," Vasil said.
</example_input>

<example_output>
Ukrainian gunners at the frontline in Donetsk region say that after months of ammunition shortages they finally have enough shells to fight with.

Demand for one hundred fifty-five millimeter artillery rounds has soared since Russia invaded Ukraine in February 2022. Allies' supplies for their own defense have been run down as they have rushed shells to Kyiv, which fires thousands of rounds per day.

(SOUNDBITE) (UKRAINIAN)  VASIL, 46, CALL SIGN SILVER, COMMANDER OF THE UNIT OPERATING M-109 HOWITZER, SAYING: 
"There was a 'shell hunger', ammunuition was rationed quite severely. It had an impact on infantry, they (Russians) crept from all sides, it hurt the infantry men. Now, there is no more 'shell hunger' and we work well."

Fresh influx of ammunition began arriving at units like Vasil's after a sixty-one billion dollar U.S. aid package was approved in April.

But most frontline units still suffer from shortages of manpower.

(SOUNDBITE) (UKRAINIAN) OLEG, 49 GUNNER SAYING: 
"We are very few, there aren't enough people. We do not even have half of the people we should have."

Outnumbered and exhausted after two and a half years in the trenches with little hope of being rotated any time soon, many are determined to keep going.

(SOUNDBITE) (UKRAINIAN)  VASIL, 46, CALL SIGN SILVER, COMMANDER OF THE UNIT OPERATING M-109 HOWITZER, SAYING: 
"we will win, we shall overcome. It is our spirit, it is our Ukraine, we are defending it. We shall overcome, at any price but we will win."
</example_output>
</example>

<example>
<example_input>
A fire at a lithium battery factory in the South Korean city of Hwaseong on Monday killed at least sixteen people. A fire official said at least five people remained missing.

Video from the scene showed rescuers wheeling out body bags from the factory and some smoke still rising from the building.

The fire, which has been largely extinguished, occurred at around ten thirty a.m. at the plant run by battery manufacturer Aricell. Local fire official Kim Jin-young said the blaze began after a series of battery cells exploded inside a warehouse containing thirty-five thousand units. He added that it remains unclear what triggered the explosion.

Yonhap news agency had earlier reported that some twenty bodies had been found inside the plant, but Kim told a televised briefing that sixteen people died and two others have suffered serious injuries.
</example_input>
<example_output>
A fire at a lithium battery factory in the South Korean city of Hwaseong on Monday killed at least sixteen people with five or more still missing.

Video from the scene showed rescuers wheeling out body bags from the factory and some smoke still rising from the building.

The fire, which has been largely extinguished, occurred at around ten thirty a.m. at the plant run by battery manufacturer Aricell. Local fire official Kim Jin-young said the blaze began after a series of battery cells exploded inside a warehouse containing thirty-five thousand units. He added that it remains unclear what triggered the explosion.
</example_output>
</example>

<example>
<example_input>
Medics in Gaza said on Monday they were working to step up screening of young children for severe malnutrition amid fears that hunger is spreading as people flee to new areas.

Aid group International Medical Corps, or IMC, and partners are planning to reach more than two hundred thousand children under five years old as part of a 'Find and Treat' campaign. One of its doctors, Mumawwar Said, told Reuters by phone.

Over the weekend, families were already coming into an IMC clinic in the central city of Deir al-Balah, opened after the agency said it had to shut down two centers in the southern city of Rafah due to insecurity.

Five-year-old Jana Ayad had weighed just nine kilograms when she arrived, suffering from diarrhea and vomiting. Nutrition Officer Raghda Ibrahim Qeshta told Reuters as she carefully held the child.

(SOUNDBITE) (Arabic) MOTHER OF FIVE-YEAR-OLD JANA AYAD, NASMA AYAD, SAYING: 
"My daughter was dying in front of me. Day after day, she was dying, her condition was very bad. Her weight dropped suddenly. I didn't know what to do."

Nasma Ayad said as she sat next to the bed, "My daughter was dying in front of me. I didn't know what to do."

Jana had started putting on some weight after treatment, medics said, but she was still painfully thin with her ribs showing as she lay listlessly in her bunny pajamas.

Staff can gauge nutrition levels by measuring the circumference of children's arms. During a Reuters cameraman's short visit at least two of the measurements were in the yellow band, indicating a risk of malnutrition.

NUTRITION OFFICER, RAGHDA IBRAHIM QESHTA, HOLDING THE MID-UPPER ARM CIRCUMFERENCE TOOL, AND SAYING (English): 'We use the (Mid-Upper Arm Circumference) MUAC. The MUAC is a tool.'

A group of U.N.-led aid agencies estimates that around seven percent of Gazan children may be acutely malnourished, compared with zero point eight percent before the Israel-Hamas conflict began on October 7th.

Until now the worst of severe hunger has been in the north, with a U.N.-backed report warning of imminent famine in March.

But aid workers worry it could spread to central and southern areas due to the upheaval around Rafah that has displaced more than one million people and constrained supply flows through southern corridors.

(SOUNDBITE) (English) NUTRITION OFFICER, RAGHDA IBRAHIM QESHTA, SAYING:
"Before the war and the conflict, the GAM or the Global Acute Malnutrition rate was 0.8% and after the conflict and the on-going conflict, and in April, the last data was that the GAM rate in Rafah increased to be 7% and in the north it is as high as 16%."

Israel launched its military operation in Gaza after Hamas-led militants attacked Israel on October 7th, killing one thousand two hundred people and taking some two hundred fifty hostage, according to Israeli tallies.

Israel says it has expanded efforts to facilitate aid flows into Gaza and blames international aid agencies for distribution problems inside the enclave.
</example_input>
<example_output>
Medics in Gaza said on Monday they were working to step up screening of young children for severe malnutrition amid fears that hunger is spreading as people flee to new areas.

Aid group International Medical Corps, or IMC, and partners are planning to reach more than two hundred thousand children under five years old as part of a 'Find and Treat' campaign.

Over the weekend, families were already coming into an IMC clinic in the central city of Deir al-Balah, opened after the agency said it had to shut down two centers in the southern city of Rafah due to insecurity.

Five-year-old Jana Ayad had weighed just nine kilograms when she arrived, suffering from diarrhea and vomiting.

(SOUNDBITE) (Arabic) MOTHER OF FIVE-YEAR-OLD JANA AYAD, NASMA AYAD, SAYING: 
"My daughter was dying in front of me. Day after day, she was dying, her condition was very bad. Her weight dropped suddenly. I didn't know what to do."

Jana had started putting on some weight after treatment, medics said, but she was still painfully thin with her ribs showing as she lay listlessly in her bunny pajamas.

A group of U.N.-led aid agencies estimates that around seven percent of Gazan children may be acutely malnourished, compared with zero point eight percent before the Israel-Hamas conflict began on October 7th.

Until now the worst of severe hunger has been in the north, with a U.N.-backed report warning of imminent famine in March.

But aid workers worry it could spread to central and southern areas due to the upheaval around Rafah that has displaced more than one million people and constrained supply flows through southern corridors.

Israel launched its military operation in Gaza after Hamas-led militants attacked Israel on October 7th, killing one thousand two hundred people and taking some two hundred fifty hostage, according to Israeli tallies.

Israel says it has expanded efforts to facilitate aid flows into Gaza and blames international aid agencies for distribution problems inside the enclave.
</example_output>
</example>

Start by brainstorming changes inside <scratchpad> tags.

Write a first draft inside <draft> tags.

Critique your draft inside <critique> tags.

KEEP WRITING DRAFTS AND CRITIQUES UNTIL THE CRITIQUE IS SATISFACTORY.

Put your final script response inside <response> tags."""
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
 - For quotes, always remove or replace parentheses/brackets and the contained text. For example this should be fully removed: (local time / 0200EST). Acronyms can be replaced such as: Korean Demilitarized Zone (DMZ) -> Korean Demilitarized Zone or DMZ. Ensure the wording flows when spoken aloud. 100,000 rupees ($358) -> One hundred thousand rupees, or three hundred fifty eight dollars.
 - For quotes, remove ALL lone hyphens (Em Dashes). Don't remove hyphens that connect two words.
 - For quotes, convert ALL numbers to spoken word. For example, 1,200 to One thousand two hundred.
 - For quotes, ALL dates need to be formatted so they can be read aloud. For example, May 1st instead of May 1.
 - Ensure double quotes in strings are replaced with single quotes to ensure JSON is valid.
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

Ensure proper JSON, replacing double quotes inside strings with single quotes. (Key/Values should still be double quotes.) Enclose your final JSON response inside <response></response> tags."""
)

headline_prompt = PromptTemplate.from_template(
"""Here is the script for a news story:

<script>
{SCRIPT}
</script>

<original_headline>
{ORIGINAL_HEADLINE}
</original_headline>

Please read the story script carefully. Then, in a <brainstorming> tag, come up with 3-5 potential headlines for this story. The headlines should:
- Summarize the key points of the story
- Be concise (MUST BE between 35 and 45 characters including spaces), and content dense
- Be written in an attention-grabbing style that would make readers want to click on the story
- Is informative over all else
- Be unique to the specifics of this story ('Missiles fired in Cuba' is better than 'Fighting in Cuba')
- Be in title case
- Spell out numbers (Two instead of 2)

Count each character count while brainstorming potential headlines. Then select the one you think is best and output it in <response></response> tags."""
)

broll_prompt = PromptTemplate.from_template(
"""You are a news video editor tasked with editing together a news video.

Here are B-roll clips:
<broll_descriptions>
{BROLL_DESCRIPTIONS}
</broll_descriptions>

Here are the section timings for the audio:
<section_timings>
{SECTION_TIMINGS}
</section_timings>

Your task is to select and place either B-roll clips or Anchor segments to accompany the audio.

<example>
**Section 1: 0.00 - 11.96**
Transcript: Activists in Canada...
Thoughts: This section is about the protests in Canada. We should show footage of the protests and the activists involved. We should cut at 7.56 to transition to the fireworks...

* **Anchor (max 10 seconds):** 0.00 - 6.24 - We start on an Anchor to introduce the story and set the scene.
* **Clip 008 (max 10 seconds):** 6.24 - 7.56 - The image of a national flag acts as a transition.
* **Clip 020 (max 12 seconds):** 7.56 - 9.82 - This clip shows fireworks being launched, visually illustrating the audio description of fireworks.
* **Clip 002 (max 4 seconds):** 9.82 - 11.96 - End the section with...
</example>

Some rules and tips:
- Follow the section timings exactly. Fill each section. Never go over each section.
- For each clip you use, ID & max duration exactly. Never go over the max duration.
- Copy the entire transcript for each section, and add your thoughts to plan out the section. Don't add ..., this is just for example.
- If no relevant B-roll is available, use an Anchor segment instead. Keep just B-roll or just Anchor segments for a complete thought/topic, then transition.

Start by writing a first draft of the edited video sequence inside <draft> tags. Follow the format shown in the example, for all sections of the audio clip.

Critique your draft inside <critique> tags.

KEEP WRITING DRAFTS AND CRITIQUES UNTIL THE CRITIQUE IS SATISFACTORY. Usually 2-3 times. Do this now, don't ask to continue.

Put your final edited video sequence inside <response> tags. YOU MUST ALWAYS END WITH THIS.
"""
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

Parse the <broll_placements></broll_placements> into each <section></section>. Start each section at 0. id should not include the clip word. Put your response in <response></response> tags."""
)

fix_broll_prompt = PromptTemplate.from_template(
"""Your task is to adjust broll placements to fit my rules. Here are the current broll placements in JSON format:

<broll_placements>
{BROLL_PLACEMENTS}
</broll_placements>

Here are the section timings:

<section_timings>
{SECTION_TIMINGS}
</section_timings>

And here are the maximum durations for each broll clip:

<broll_timings>  
{BROLL_TIMINGS}
</broll_timings>

Check the broll placements and make adjustments as needed to ensure they follow these rules:

- Do not exceed any broll's maximum duration as specified in <broll_timings>
- Fill each section with broll placements until the section's end time, but do not go beyond the section end time
- Clip timings must end exactly at their containing section's end time
- You cannot use more of a broll clip than its maximum duration allows, so switch to a different clip when needed
- Place an Anchor block of anywhere between 5-10 seconds (inclusive, can just be 5 seconds) at the very beginning of the first section to set the scene
- Place an Anchor block of anywhere between 5-10 seconds (inclusive) at the very end of the last section to conclude the story 
- An Anchor block placement must be at least 3 seconds long
- Always show a broll clip for at least 1 second before switching to a different one
- Do not make up any new broll clips. If there is unfilled time, you may reuse existing clips, prioritizing reusing the Anchor clip
- Not every section needs an Anchor block

Example adjustments include:

- Broll clip was too short (< 1 second), increase its duration to 2 seconds while decreasing the broll clip before it to 3.5 seconds
- Broll clip was too short & cannot be extended. Removing the entire clip and increasing duration of surrounding broll to fill it in (ensuring these do not exceed their max duration)
- Final anchor placement is too short (< 5 seconds), increasing its duration
- Anchor placement is too short (< 3 seconds), increasing duration

First mark down the durations of every broll/anchor placement inside <durations></durations> tags.

Then mark down any violating broll/anchor placements that need to be change inside <violations></violations> tags.

After making the necessary adjustments, output the updated broll placements JSON inside <response></response> tags.
Ensure it's valid JSON using double quotes, escaping special characters, and converting double quotes in strings to single quotes. If no adjustments need to be made, output the original broll placements JSON."""
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
possible, even if some words aren't present in the quote. If there is no fuzzy match, return Unknown. Put your final response in <response></response> tags."""
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
         "sot_id": <sot_id>
      }},
      ...
   ]
}}
</example>

The transcript from a clip should match the quote from a sot to match. If a clip doesn't match any sot, set sot_id to null.
Ensure properly formatted JSON. Put your response in <response></response> tags."""
)

tts_prompt = PromptTemplate.from_template(
"""Here is some text that will be spoken aloud on television:

<script>
{SCRIPT}
</script>

This script will be spoken word for word, and therefore needs to be adjusted to be spoken naturally.

 - Always remove or replace parentheses/brackets and the contained text. For example this should be fully removed: (local time / 0200EST). Acronyms can be replaced such as: Korean Demilitarized Zone (DMZ) -> Korean Demilitarized Zone or DMZ. Ensure the wording flows when spoken aloud. 100,000 rupees ($358) -> One hundred thousand rupees, or three hundred fifty eight dollars.
 - Remove ALL lone hyphens (Em Dashes). Don't remove hyphens that connect two words.
 - Convert ALL numbers to spoken word. For example, 1,200 to One thousand two hundred.
 - ALL dates need to be formatted so they can be read aloud. For example, May 1st instead of May 1.

Don't change any of the actual words, just how they are spoken. Make any other changes that will improve how these are spoken.
Put your response in <response></response> tags.
"""
)

json_prompt = PromptTemplate.from_template(
"""Fix this JSON object to be properly formatted:

<json>
{JSON}
</json>

Ensure it's valid JSON escaping special characters and converting double quotes in strings to single quotes. (Keys/Values should still be in double quotes) Put your response in <response></response> tags."""
)

spell_check_chain = (spell_check_prompt | sonnet35).with_config({"run_name": "spell_check"})
get_sot_chain = (get_sot_prompt | sonnet35).with_config({"run_name": "get_sots"})
facts_chain = (facts_prompt | sonnet35).with_config({"run_name": "generate_facts"})
parse_sot_chain = (parse_sot_prompt | sonnet35).with_config({"run_name": "parse_sots"})
reformat_chain = (reformat_prompt | sonnet35).with_config({"run_name": "reformat_script"})
sot_chain = (sot_prompt | sonnet35).with_config({"run_name": "add_sots"})
edit_chain = (edit_prompt | sonnet35).with_config({"run_name": "edit_script"})
parse_chain = (parse_prompt | sonnet35).with_config({"run_name": "parse_script"})
logline_chain = (logline_prompt | sonnet35).with_config({"run_name": "logline"})
headline_chain = (headline_prompt | sonnet35).with_config({"run_name": "headline"})
broll_chain = (broll_prompt | sonnet35).with_config({"run_name": "broll"})
parse_broll_chain = (parse_broll_prompt | sonnet35).with_config({"run_name": "parse_broll"})
fix_broll_chain = (fix_broll_prompt | sonnet35).with_config({"run_name": "fix_broll"})
match_sot_chain = (match_sot_prompt | sonnet35).with_config({"run_name": "match_sot"})
match_hard_sot_chain = (match_hard_sot_prompt | sonnet35).with_config({"run_name": "match_hard_sot"})
language_to_iso_chain = (language_to_iso_prompt | sonnet35).with_config({"run_name": "language_to_iso"})
match_clip_to_sots_chain = (match_clip_to_sots_prompt | sonnet35).with_config({"run_name": "match_clip_to_sots"})
json_chain = (json_prompt | sonnet35).with_config({"run_name": "json"})

from sqlalchemy.exc import OperationalError
import time
from anthropic import APIError
from src.hashing import hash_chain
from langchain_core.runnables.base import RunnableBinding

def extract_response(text):
    return text[text.rfind("<response>"):text.rfind("</response>")+11]

def extract_xml(text):
    return XMLOutputParser().invoke(extract_response(text).replace("&", "and"))

@st.cache_data(show_spinner=False, hash_funcs={RunnableBinding: hash_chain})
def run_chain(chain, params, max_retries=3, retry_delay=5):
    """Runs the LangChain chain with retry logic."""
    retries = 0
    while retries <= max_retries:
        try:
            response_raw = chain.invoke(params).content
            response_xml = extract_xml(response_raw)
            return response_xml['response'].strip()
        except OperationalError:
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

@st.cache_data(show_spinner=False, hash_funcs={RunnableBinding: hash_chain})
def run_chain_json(chain, params):
    response = run_chain(chain, params)
    try:
        return JsonOutputParser().invoke(response)
    except Exception as e:
        response = run_chain(json_chain, {"JSON": response})
        return JsonOutputParser().invoke(response)
