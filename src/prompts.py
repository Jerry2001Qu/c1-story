# Prompts

# STREAMLIT
from src.constants import LANGCHAIN_API_KEY, ANTHROPIC_API_KEY
import streamlit as st
# /STREAMLIT

import os

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Channel 1"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

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
Include the entire section from the shotlist including parts like "(SOUNDBITE)(English) LOREM IPSUM, SAYING:".
Only include shots that have people speaking. Don't include shots that are text clips such as "POST ON TRUTH SOCIAL PLATFORM BY REPUBLICAN PRESIDENTIAL CANDIDATE DONALD TRUMP, READING (English):"
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

reformat_title_prompt = PromptTemplate.from_template(
"""Reformat a person's title to fit into a maximum of 7-8 words.

Here is the title:
<title>
{TITLE}
</title>

- The title should be rephrased to fit into a maximum of 7-8 words.
- If the title is already 8 words or less, do not change it.
- Do not change the meaning of the title.

Here is an example:
<example>
<title>
Chief Executive Officer and President of The National Broadcasting Company in New York City
</title>

<response>
Chief Executive Officer and President of NBC
</response>
</example>

Put your response in <response></response> tags.
"""
)

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
    - (SOUNDBITE) (English) SKY NEWS AUSTRALIA WEEKEND LIVE PRESENTER, KIERAN GILBERT, SAYING:
      "We begin this hour with breaking news, the former President Donald Trump has survived what law authorities are calling an assassination attempt at a rally."
      (This is a soundbite. Do not change it. It may be at the beginning of the story, still do not change it.)
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

The air was thick with the sound of sirens as the crowd grew increasingly agitated, resisting the police's efforts to disperse the impromptu gathering.

Fans, some equipped with concert posters and glow sticks, tried to prevent the police from entering the venue by forming human chains, all while chanting, "let us sing" and shining flashlights towards the officers.

FAN EXCLAIMING (English) "I'VE WAITED MONTHS FOR THIS, YOU CAN'T JUST TAKE IT AWAY" WHILE BEING ESCORTED OUT BY POLICE

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

In a <scratchpad>, first count the number of sentences in the script & divide by 3 to get the maximum number of quotes.
Then order the quotations based on how well they'd fit into the script & plan the placement of the top quotes. Remember to:

- Keep the quotations short, extracting just a portion of the original quote if needed to keep them concise. No more than 1 or 2 sentences.
- If extracting a portion of a quotation, ensure the portion is CONTIGUOUS. Never cut a gap out of the portion.
- Only insert quotations that are relevant, add value, and are interesting to the viewer. If it's not interesting, don't insert it.
- Spread the quotations throughout the story
- Don't add too many quotations. Maximum 1 every 3 sentences. You do not have to use all of the quotations.
- You may start the story with an English quotation if it makes sense to do so.
  - For example, you could start with a quote of protestors chanting.
  - Only start stories with short quotes, maximum 5 words. If longer, you may cut it down.
  - An example could be: "NO JUSTICE, NO PEACE!" chanted the crowd as they marched through the streets.
- NEVER start with a quote in a language other than English.
- NEVER put a quotation at the end of the entire story. We must end with the script
- Insert the portions of quotations including their descriptions
- Do not add any transitions between the quotations and the rest of the script
- Do not change the wording of the news script except to insert the quotations
- Place quotations in separate paragraphs. You may split paragraphs in the original script.

Here are some general criteria of good soundbites:
- Short soundbites such as chanting or short exclamations. They're good as transitions and adding real voices to the story.
- Soundbites that reflect what was said in the script.
- Soundbites that are directly referenced in the script.
- Key events referenced in the script.

After planning it out, provide your final response with the quotations inserted into the appropriate parts of the news script inside <response></response> tags.
"""
)

edit_prompt = PromptTemplate.from_template(
"""I'm producing a television news segment. You are an expert editor which will edit my news script.

Here is the original script:
<script>
{SCRIPT}
</script>

Here are available soundbites:
<soundbites>
{SOUNDBITES}
</soundbites>

You will edit this script to maintain a proper 'flow', while removing unnecessary/irrelavent information.

You may only make simple edits, including:
- Reordering sections
- Removing sections
- Grammatical changes that don't change meaning
  - But like most frontline units, Vasil's team suffers from shortages of manpower. -> But most frontline units still suffer from shortages of manpower.

- Don't change the meaning of any sentences. Never state what someone else said as your own.
- The anchor should not give opinions. Opinions can be in quotes/statements from other individuals.
  - NEVER add your own opinions or agree/disagree with the content. You just report what happened.
- Ensure your changes are factual & do not misrepresent the original script.
- Remember the current date. Past events should be in past tense.

The script may contain soundbites. They are in a format like:
(SOUNDBITE) (Urdu) LOCAL RESIDENT, MOHAMMAD IMRAN, SAYING:
"People in Karachi are facing extreme heat at the moment. Look..."

- Do not remove any formatting around soundbites. Leave the "(SOUNDBITE) (Urdu) LOCAL RESIDENT, MOHAMMAD IMRAN, SAYING:"
- You may add, reorder, remove, or cut down soundbites.
- If cutting down a soundbite, ensure the soundbite is CONTIGUOUS. Therefore, you may only cut from the start or end of a soundbite!
- Soundbites are better when spread throughout the story instead of clumped together.
- Short soundbites are good. Chanting or short exclamations should be kept in the story to add real voices to the story.
- Don't add too many quotations. Maximum 1 every 3 sentences. You do not have to use all of the quotations.
- You may start the story with an English quotation if it makes sense to do so.
  - For example, you could start with a quote of protestors chanting.
  - Only start stories with short quotes, maximum 5 words. If longer, you may cut it down.
  - An example could be: "NO JUSTICE, NO PEACE!" chanted the crowd as they marched through the streets.
- NEVER start with a quote in a language other than English.

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
- Be concise (MUST BE between 35 and 44 characters including spaces), and content dense
- Be written in an attention-grabbing style that would make readers want to click on the story
- Is informative over all else
- Uses simple language for a general audience.
- Be unique to the specifics of this story ('Missiles fired in Cuba' is better than 'Fighting in Cuba')
- Be in title case
- Spell out single numbers (Two instead of 2)

Here are some examples of good headlines:
<good_headlines>
- New York City Announces Free Metro Passes
- 100,000 People Attend Climate Change Rally in London
- Alex Baldwin 'Rust' Shooting: Trial Begins
</good_headlines>

Count each character count while brainstorming potential headlines. Then select the one you think is best and output it in <response></response> tags."""
)

broll_request_prompt = PromptTemplate.from_template(
"""You are a news video editor tasked with finding relevant B-roll clips for a news story.

Here is the script:
<script>
{SCRIPT}
</script>

Here is an example:
<example>
Activists in Canada are protesting against the government's new policies. The protests have been ongoing for weeks, with activists demanding change. The situation escalated last night when police clashed with protesters, leading to several arrests.

Ideal:
- Protesters marching with signs
- Police arresting protesters
- Activists chanting slogans
- Aerial view of the protest

Secondary:
- Police cars arriving at the scene
- Any shots of the protesters
- Footage of the government building

Acceptable:
- Still images of the protest
- Stock footage of protests
- Any relevant footage of Canada
</example>

Please create a list of B-roll clips that would be relevant to this story.
- Each B-roll clip should be described in a single sentence.
- Create 3 lists going from most to least relevant B-roll clips.
- The last list should continue very general B-roll clips that could be used if more specific clips are not available.

Put your response in <response></response> tags."""
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
- Copy import parts of the transcript. Add your thoughts to plan out the section.
- If no relevant B-roll is available, use an Anchor segment instead. Keep just B-roll or just Anchor segments for a complete thought/topic, then transition.

Start by writing a first draft of the edited video sequence inside <draft> tags. Follow the format shown in the example, for all sections of the audio clip.

Critique your draft inside <critique> tags.

KEEP WRITING DRAFTS AND CRITIQUES UNTIL THE CRITIQUE IS SATISFACTORY. Usually 1-2 times. Do this now, don't ask to continue.

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
            "id": "011_0",
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

courtesy_prompt = PromptTemplate.from_template(
"""Identify any courtesy requirements in the shotlist and put them in JSON.

Here's an example:
<example>
VIDEO SHOWS: SECURITY PERSONNEL SCUFFLING WITH PEOPLE NEAR STADIUM GATES / MAN WITH HANDS IN ZIP TIES LYING ON GROUND, BEING REMOVED BY SECURITY

RESENDING WITH COMPLETE SCRIPT

Verified by:

- Original file metadata

SHOWS: MIAMI GARDENS, FLORIDA, UNITED STATES (JULY 14, 2024) (SAMMBONU - No resale / Must on-screen courtesy sammbonu)

1. SECURITY PERSONNEL SCUFFLING WITH FAN

2. MAN WITH HANDS IN ZIP TIES LYING ON GROUND, SECURITY PERSONNEL HELPING HIM STAND AND REMOVING HIM FROM PREMISES

STORY: Security scuffled with fans after organizers said thousands of supporters without tickets rushed security and tried to force their way into the stadium, delaying the Copa America final between Argentina and Colombia by more than hour on Sunday (July 14).

Video obtained by Reuters showed security manhandling fans near the gates of the Hard Rock Stadium in Miami Gardens, Florida. Footage also showed a man lying on the ground with his hands behind him in zip ties before being removed by security from the premises.

The date and location of the footage was verified by corroborating visuals of the event shot by different sources and original file metadata from the source.

Police managed to close the gates and initiate a lockdown, leading to scores of people stuck outside trying to enter before the match kicked off.

South American football's governing body CONMEBOL pushed the match's start time back three times from 8:00 p.m. to 8:30 p.m., to 8:45 p.m. and finally to 9:15 p.m, when the teams were finally able to line up for the national anthems.

<example_response>
{{
   "clips": [
      {{
         "id": 1,
         "courtesy": "sammbonu"
      }},
      {{
         "id": 2,
         "courtesy": "sammbonu"
      }}
   ]
}}
</example_response>
</example>

<example>
VIDEO SHOWS: VARIOUS OF LOCATION HALF A MILE FROM SITE OF TRUMP PENNSYLVANIA RALLY SHOOTING.

SHOTLIST ONLY; FULL SCRIPT TO FOLLOW

SHOWS: NEAR BUTLER, PENNSYLVANIA, UNITED STATES (JULY 14, 2024) (REUTERS - Access all)

1. VARIOUS OF POLICE CAR WITH TRAFFIC FLOWING INFRONT OF 'ROAD CLOSED' CORDONS

2. MEDIA AT SITE

SHOWS: BUTLER, PENNSYLVANIA, UNITED STATES (JULY 14, 2024) (US TV - Must on screen courtesy 'US TV')

3. SIGNS READING (English): 'TRUMP GEAR HERE' AND 'YARD SALE '

4. VARIOUS OF POLICE CAR WITH TRAFFIC FLOWING INFRONT OF 'ROAD CLOSED' CORDONS

STORY: Police barriers were set up on Sunday (14 July) near the site of the Donald Trump Pennsylvania rally where a gunman took shots at the Republican presidential hopeful.

The site, half a mile from the location of the shooting in Butler, Pennsylvania, has been cordoned off on Sunday.

The FBI identified 20-year-old Thomas Matthew Crooks of Bethel Park, Pennsylvania as the suspect in Saturday's attempted assassination of former U.S. President Donald Trump at a campaign rally.

The suspect was shot and killed by the Secret Service seconds after he allegedly fired shots toward a stage where Trump was speaking.

The FBI said it was working to determine a motive for the attack, in which one rally attendee died and two other spectators were critically injured. Trump was shot in the ear.

<example_response>
{{
   "clips": [
      {{
         "id": 1,
         "courtesy": 0
      }},
      {{
         "id": 2,
         "courtesy": 0
      }},
      {{
         "id": 3,
         "courtesy": "US TV"
      }},
      {{
         "id": 4,
         "courtesy": "US TV"
      }}
   ]
}}
</example_response>
</example>

<example>
SHOWS: BALTIMORE, MARYLAND, UNITED STATES (JUNE 24, 2024) (ABC AFFILIATE WJLA - Broadcast: No use USA. No archive. Digital: No use USA. No archive.)

1. AERIAL OF CONTAINER SHIP DALI BEING TOWED, PASSING BETWEEN THE SUPPORTS OF THE FRANCIS SCOTT KEY BRIDGE

2. AERIAL SHOWING CLOSE-UP OF THE FRONT OF CONTAINER SHIP DALI WITH DEBRIS AND DAMAGED CONTAINERS

3. AERIAL OF DALI IN HARBOUR

4. AERIAL OF DALI APPROACHING BRIDGE

5. AERIAL OF SIDE VIEW OF DALI

6. VARIOUS AERIALS OF DALI PASSING BETWEEN THE SUPPORTS OF THE FRANCIS SCOTT KEY BRIDGE

<example_response>
{{
   "clips": [
      {{
         "id": 1,
         "courtesy": 0
      }},
      {{
         "id": 2,
         "courtesy": 0
      }},
      {{
         "id": 3,
         "courtesy": 0
      }},
      {{
         "id": 4,
         "courtesy": 0
      }},
      {{
         "id": 5,
         "courtesy": 0
      }},
      {{
         "id": 6,
         "courtesy": 0
      }}
   ]
}}
</example_response>
</example>

Here's the shotlist and surrounding text:
<shotlist>
{BODY}
</shotlist>

Identify any courtesy requirements in the shotlist and put them in JSON. If there are no courtesy requirements for a clip, set courtesy to 0. Always do this for every clip id. Put your JSON response in <response></response> tags.
"""
)

extract_storyline_and_shotlist_prompt = PromptTemplate.from_template(
"""Extract the storyline and shotlist from the following text.

Here is an example:
<example>
<example_input>
VIDEO SHOWS: SECOND WRAP OF EDITS FOLLOWING SHOOTING INCIDENT AT FORMER U.S. PRESIDENT DONALD TRUMP'S RALLY IN PENNSYLVANIA

EDITORS PLEASE NOTE: THIS EDIT IS A WRAP AND CONTAINS NO FRESH MATERIAL / ORIGINAL EDIT NUMBERS AND STORY DETAILS INCLUDED IN SHOTLIST

RESENDING WITH COMPLETE SCRIPT

SHOWS: Eyewitness footage at Donald Trump's campaign rally on Saturday (July 13) showed people shouting and panicking during a shooting attack that left the presidential candidate's face streaked with blood. Reuters corroborated visuals from the scene such as the grey structure with glass windows also seen in the video by comparing it to matching file imagery. The date of the video matched that of Reuters and other media reports. (8706-USA-ELECTION/TRUMP-UGC2)

BUTLER, PENNSYLVANIA, UNITED STATES (JULY 13, 2024) (VIDEO OBTAINED BY REUTERS VIA TMX - No resale / Must on-screen courtesy VIDEO OBTAINED BY REUTERS VIA TMX) (ORIGINALLY SHOT IN PORTRAIT)

1. STAGE AT RALLY / PEOPLE PANICKING AS AUDIO HEARD SHOUTING (English): "He's got a gun!" / AUDIO OF GUNFIRE / GREY STRUCTURE / VOICE HEARD SAYING (English): "He's on top of the roof, don't go over there. He's on the roof buddy." / PEOPLE WALKING AROUND / VOICE HEARD SAYING (English): "I can see his head blow up, they shot him in the head" (PROFANITY HEARD IN BACKGROUND) / VOICE HEARD SAYING (English): "He's pulled off a number of shots. The first few number of shots were him." / PEOPLE WALKING AND GREY STRUCTURE / POLICE MAN WALKING

A 2020 high school yearbook photo of Thomas Matthew Crooks of Bethel Park, Pennsylvania, was released on Sunday (July 14) after the FBI identified the 20-year-old as the suspect involved in the attempted assassination of former U.S. President Donald Trump at a campaign rally. (8824-USA-ELECTION/TRUMP-SHOOTER-STILLS)

BETHEL PARK, PENNSYLVANIA, UNITED STATES (FILE) (OBTAINED BY REUTERS - No Archive/No Resale) (MUTE)

2. VARIOUS STILL PHOTOGRAPHS OF 2020 HIGH SCHOOL YEARBOOK PHOTO OF THOMAS MATTHEW CROOKS, NAMED BY FBI AS 'SUBJECT INVOLVED' IN ATTEMPTED ASSASSINATION OF FORMER U.S. PRESIDENT DONALD TRUMP

An eyewitness recalled on Sunday (July 14) seeing a man with a riffle on a nearby roof moments before former US president Donald Trump was shot in the ear at a campaign rally in Pennsylvania. Greg Smith said he had alerted police and Secret Services just before shots were heard. (8813-USA-ELECTION/TRUMP-SHOOTING-EYEWITNESS)

BUTLER, PENNSYLVANIA, UNITED STATES (JULY 13, 2024) (BBC - Mandatory onscreen credit BBC News/ No use after 1500GMT on July 16, 2024)

3. (SOUNDBITE) (English) EYEWITNESS AND TRUMP SUPPORTER GREG SMITH, SPEAKING TO REPORTER, SAYING:

"We noticed a guy crawling, bear-crawling up the roof of the building beside us, 50 feet away from us. So we're standing there, we're pointing at the guy crawling up the roof."

REPORTER: 'And he had a gun right.'

SMITH: 'A rifle, we can clearly see him with a rifle, absolutely. We're pointing at him, the police are down there running around on the ground, we're like, hey man, there's a guy on the roof with a rifle. And the police are like, huh, what, they didn't know what was going on, we're like right here on the roof, we can see him from right here, we see him, he's crawling. And next thing you know, I'm thinking to myself, why is Trump still speaking, why have they not pulled him off the stage. I'm standing there pointing at him for two or three minutes. Secret services are looking at us from the top of the barn, I'm pointing at that roof, just standing there like this, and next thing you know five shots ring out."

Witnesses who attended the former U.S. President Donald Trump’s rally in Butler, Pennsylvania described the attempted assassination on Sunday (July 14). (8806-USA-ELECTION/TRUMP-SHOOTING-BUTLER REACTION)

BUTLER, PENNSYLVANIA, UNITED STATES (JULY 14, 2024) (REUTERS - Access all)

4. (SOUNDBITE) (English) TRUMP SUPPORTER FROM FROM NEWLAND, NORTH CAROLINA, RENEE WHITE, SAYING:

“So, I heard the first shot. And as soon as I heard that first shot, I turned over and I looked at Trump and then I heard two more. It was like, bang, bang, bang. It's like three right in a row. I looked at Trump and I saw him looking and I saw him drop to the ground. And then as he, as he dropped to the ground, then I saw the Secret Service come around him. I saw the guys and all the military with the guns and everything come around him. I heard all the calls they were doing. I heard what they were saying back and forth about, you know, waiting until they had the all clear that they had everything."

Police barriers were set up on Sunday (14 July) near the site of the Donald Trump Pennsylvania rally where a gunman took shots at the Republican presidential hopeful. (8770-USA-ELECTION/TRUMP-SHOOTING SITE-MORNING)

NEAR BUTLER, PENNSYLVANIA, UNITED STATES (JULY 14, 2024) (REUTERS - Access all)

5. VARIOUS OF POLICE CAR WITH TRAFFIC FLOWING IN FRONT OF 'ROAD CLOSED' CORDONS

6. MEDIA AT SITE

7. SIGNS READING (English): 'TRUMP GEAR HERE' AND 'YARD SALE '

A local district attorney in Butler, Pennsylvania, expressed embarrassment on Sunday (July 14) following the assassination attempt on former U.S. President Donald Trump the day before. (8797-USA-ELECTION/TRUMP-SHOOTING-LOCAL PROSECUTOR)

BUTLER, PENNSYLVANIA, UNITED STATES (JULY 14, 2024) (REUTERS - Access all)

8. (SOUNDBITE) (English) BUTLER COUNTY DISTRICT ATTORNEY, RICHARD GOLDINGER, SAYING:

“I've lived here my whole life and you just don't think that something like this would happen in your hometown. We haven't had an attempt on a, on a president assassination attempt since Reagan in 1980. Now, for it to happen in your backyard, you just don't expect that. So it's embarrassing for the community, but at the same time, I mean, you just can't predict, you know, human behaviour sometimes.”

9. VARIOUS OF POLICE OFFICERS AND VEHICLE NEAR SITE OF TRUMP RALLY IN BUTLER, PENNSYLVANIA

Thomas Crooks, the man named by the FBI as the shooter in the attempted assassination of Donald Trump, was "bullied every day" in high school, a high school classmate told media on Sunday (July 14). Jason Kohler, who graduated from Bethel Park High School the year before Crooks, described him as an "outcast" who was "always alone". Kohler said he did not have any interactions with Crooks but spoke of him sitting alone at lunch and wearing hunting outfits to school. The 21-year-old local resident also said he thought Crooks had been on the school's rifle team, though he was uncertain. (8821-USA-ELECTION/TRUMP-BETHEL-PARK-SCHOOL CLASSMATE)

BETHEL PARK, PENNSYLVANIA, UNITED STATES (JULY 14, 2024) (REUTERS - Access all)

10. (SOUNDBITE) (English) CLASSMATE OF THOMAS CROOKS AT BETHEL PARK HIGH SCHOOL, JASON KOHLER, SAYING:

"I didn't have any interaction with him, but he was, like a kid that was always alone. He was always bullied. Every day. He was just an outcast. Yeah."

The neighbourhood of Bethel Park, Pennsylvania, was abuzz with media and police on Sunday (July 14) a day after one of its residents allegedly tried to kill former U.S. President Donald Trump at a campaign rally about one hour away in the city of Butler. (8791-USA-ELECTION/TRUMP-SHOOTER-BETHEL-PARK)

BETHEL PARK, PENNSYLVANIA, UNITED STATES (JULY 14, 2024) (REUTERS - Access all)

11. VARIOUS OF POLICE OFFICERS STANDING NEAR VEHICLES IN NEIGHBOURHOOD WHERE SUSPECTED TRUMP RALLY SHOOTER THOMAS MATTHEW CROOKS LIVED

12. VARIOUS OF NEWS CAMERAS

13. VARIOUS OF PENNSYLVANIA STATE POLICE VEHICLE

Residents of the Pittsburgh suburb of Bethel Park expressed shock and sadness on Sunday (July 14) after finding out a local man had been identified as the shooter in an attempted assassination of Donald Trump. (8802-USA-ELECTION/TRUMP-BETHEL-PARK-VOXIES)

BETHEL PARK, PENNSYLVANIA, UNITED STATES (JULY 14, 2024) (REUTERS - Access all)

14. (SOUNDBITE) (English) RETIRED RESIDENT, JIM ZAWOJSKI, 70, SAYING:

"I was stunned. I mean, where is America going to now? Okay. I was talking to one other reporter, and I said, the country's been never so divided. Okay? Everyone needs to go and take a step back. Relax. If you have differences with a candidate, you do it at the voting booth. You know? You don't do it... you don't take action like this clown did. Okay? If it happened to Biden, I'd be just as stunned, you know, because there is no need for that in our society like this."

Supporters of former U.S. President Donald Trump gathered outside Trump Tower in New York on Sunday (July 14), after he survived an assassination attempt days before he was due to accept the formal Republican presidential nomination. (8831-USA-ELECTION/TRUMP-SHOOTING-TOWER SUPPORTERS and 8814-USA-ELECTION/TRUMP-SHOOTING-TRUMP TOWER REACTION)

NEW YORK, NEW YORK, UNITED STATES (JULY 14, 2024) (REUTERS - Access all)

15. VARIOUS OF TRUMP SUPPORTERS STANDING ON PICKUP TRUCK AND WAVING FLAGS, TRUMP TOWER SEEN IN BACKGROUND

16. VARIOUS OF PICKUP TRUCK PARKED OUTSIDE TRUMP TOWER WITH SUPPORTERS ON BACK WAVING FLAG

17. FLAG WITH ILLUSTRATION OF TRUMP’S FACE ON IT

18. (SOUNDBITE) (English) CALIFORNIA RESIDENT, ARMIN (NO LAST NAME GIVEN), SAYING:

“You know what happened yesterday was a tragedy for this country. It's unthinkable that, you know, something like this can happen. And, you know, I pray for Trump. You know, he's he's trying to save our country from, you know, the radicals that exist. He wants a better USA for us all. And there's now people are trying to assassinate him, which is an unthinkable thing. And, you know, he doesn't have to do this job. He's 78 years old. He should be relaxing, golfing. But he's risking his life, you know, to save the country.”

19. PEOPLE ENTERING TRUMP TOWER

20. ENTRANCE OF TRUMP TOWER

21. TRUMP SUPPORTER HOLDING BANNER, READING (English): 'PURSUED, PERSECUTED, PROSECUTED'

It was a quiet scene near the former U.S. President Donald Trump’s property in Bedminster, New Jersey on Sunday (July 14). (8778-USA-ELECTION/TRUMP-SHOOTING-BEDMINSTER-MORNING)

BEDMINSTER, NEW JERSEY, UNITED STATES (JULY 14, 2024) (REUTERS - Access all)

22. STREET NEAR TRUMP’S GOLF CLUB PROPERTY IN BEDMINSTER, NEW JERSEY

23. CAMERA EQUIPMENT / REPORTERS AND VEHICLES SEEN FROM ACROSS THE STREET

24. STREET SIGN READING (English) ‘CLUCAS BROOK RD’

25. SIGNS READING (English) ‘SOMERSET 523 COUNTY’ AND ‘MILE 29’

In a country already on edge, the assassination attempt on former President Donald Trump has enraged his supporters, paused the Democratic campaign, and raised fears of further political violence in the run-up to November's election. According to analysts, the country is at a tipping point. (8807-USA-ELECTION/TRUMP-SHOOTING-ANALYST)

CHICAGO, ILLINOIS, UNITED STATES (JULY 14, 2024) (REUTERS - Access all)

26. (SOUNDBITE) (English) UNIVERSITY OF CHICAGO PROFESSOR IN THE POLITICAL SCIENCE DEPARTMENT AND THE CHICAGO PROJECT ON SECURITY AND THREATS DIRECTOR, ROBERT PAPE, SAYING:

"Support for political violence matters because that is what normalizes political violence for volatile actors who would go that next step. So it's often the case that the actors who are the lone wolves are, I call them, 'volatile' because they can be mentally ill. They can be inundated with propaganda coming from one of the political parties and so forth. And then and then the media will say, 'Oh, well, that's not really political violence or something.' No, these are not competing explanations."

U.S. President Joe Biden delivered remarks on Sunday (July 14), one day following the assassination attempt on former President Donald Trump. (8827-USA-ELECTION/TRUMP-SHOOTING-BIDEN)

WASHINGTON, D.C., UNITED STATES (JULY 14, 2024) (US NETWORK POOL - Broadcast: No use USA. Digital: No use USA.)

27. (SOUNDBITE) (English) U.S. PRESIDENT, JOE BIDEN, SAYING:

“Last night, I spoke with Donald Trump. I'm sincerely grateful that he's doing well and recovering."

28. (SOUNDBITE) (English) U.S. PRESIDENT, JOE BIDEN, SAYING:

"Mr. Trump, as a former president, the nominee of the Republican Party already receives a heightened level of security. And I've been consistent in my direction to the Secret Service to provide him with every resource, capability, and protective measure necessary to ensure his continued safety."

Former U.S. President Donald Trump thanked well-wishers on Sunday (July 14) on his social media website Truth Social, after he was shot in the ear in an attempted assassination during a campaign rally. (8780-USA-ELECTION/TRUMP-TRUTH SOCIAL UPDATE)

INTERNET (JULY 14, 2024) (SOCIAL MEDIA WEBSITE - Access all) (MUTE)

29. SCREENSHOT FROM TRUMP'S SOCIAL MEDIA ACCOUNT 'TRUTH SOCIAL' READING (English):

"Thank you to everyone for your thoughts and prayers yesterday, as it was God alone who prevented the unthinkable from happening. We will FEAR NOT, but instead remain resilient in our Faith and Defiant in the face of Wickedness. Our love goes out to the other victims and their families. We pray for the recovery of those who were wounded, and hold in our hearts the memory of the citizen who was so horribly killed. In this moment, it is more important than ever that we stand United, and show our True Character as Americans, remaining Strong and Determined, and not allowing Evil to Win. I truly love our Country, and love you all, and look forward to speaking to our Great Nation this week from Wisconsin. DJT"

STORY: Please see shotlist for story details.

(Production: Lynné Schoeman)
</example_input>

<example_response>
<shotlist>
1. STAGE AT RALLY / PEOPLE PANICKING AS AUDIO HEARD SHOUTING (English): "He's got a gun!" / AUDIO OF GUNFIRE / GREY STRUCTURE / VOICE HEARD SAYING (English): "He's on top of the roof, don't go over there. He's on the roof buddy." / PEOPLE WALKING AROUND / VOICE HEARD SAYING (English): "I can see his head blow up, they shot him in the head" (PROFANITY HEARD IN BACKGROUND) / VOICE HEARD SAYING (English): "He's pulled off a number of shots. The first few number of shots were him." / PEOPLE WALKING AND GREY STRUCTURE / POLICE MAN WALKING

2. VARIOUS STILL PHOTOGRAPHS OF 2020 HIGH SCHOOL YEARBOOK PHOTO OF THOMAS MATTHEW CROOKS, NAMED BY FBI AS 'SUBJECT INVOLVED' IN ATTEMPTED ASSASSINATION OF FORMER U.S. PRESIDENT DONALD TRUMP

3. (SOUNDBITE) (English) EYEWITNESS AND TRUMP SUPPORTER GREG SMITH, SPEAKING TO REPORTER, SAYING:

"We noticed a guy crawling, bear-crawling up the roof of the building beside us, 50 feet away from us. So we're standing there, we're pointing at the guy crawling up the roof."

REPORTER: 'And he had a gun right.'

SMITH: 'A rifle, we can clearly see him with a rifle, absolutely. We're pointing at him, the police are down there running around on the ground, we're like, hey man, there's a guy on the roof with a rifle. And the police are like, huh, what, they didn't know what was going on, we're like right here on the roof, we can see him from right here, we see him, he's crawling. And next thing you know, I'm thinking to myself, why is Trump still speaking, why have they not pulled him off the stage. I'm standing there pointing at him for two or three minutes. Secret services are looking at us from the top of the barn, I'm pointing at that roof, just standing there like this, and next thing you know five shots ring out."

4. (SOUNDBITE) (English) TRUMP SUPPORTER FROM FROM NEWLAND, NORTH CAROLINA, RENEE WHITE, SAYING:

“So, I heard the first shot. And as soon as I heard that first shot, I turned over and I looked at Trump and then I heard two more. It was like, bang, bang, bang. It's like three right in a row. I looked at Trump and I saw him looking and I saw him drop to the ground. And then as he, as he dropped to the ground, then I saw the Secret Service come around him. I saw the guys and all the military with the guns and everything come around him. I heard all the calls they were doing. I heard what they were saying back and forth about, you know, waiting until they had the all clear that they had everything."

5. VARIOUS OF POLICE CAR WITH TRAFFIC FLOWING IN FRONT OF 'ROAD CLOSED' CORDONS

6. MEDIA AT SITE

7. SIGNS READING (English): 'TRUMP GEAR HERE' AND 'YARD SALE '

8. (SOUNDBITE) (English) BUTLER COUNTY DISTRICT ATTORNEY, RICHARD GOLDINGER, SAYING:

“I've lived here my whole life and you just don't think that something like this would happen in your hometown. We haven't had an attempt on a, on a president assassination attempt since Reagan in 1980. Now, for it to happen in your backyard, you just don't expect that. So it's embarrassing for the community, but at the same time, I mean, you just can't predict, you know, human behaviour sometimes.”

9. VARIOUS OF POLICE OFFICERS AND VEHICLE NEAR SITE OF TRUMP RALLY IN BUTLER, PENNSYLVANIA

10. (SOUNDBITE) (English) CLASSMATE OF THOMAS CROOKS AT BETHEL PARK HIGH SCHOOL, JASON KOHLER, SAYING:

"I didn't have any interaction with him, but he was, like a kid that was always alone. He was always bullied. Every day. He was just an outcast. Yeah."

11. VARIOUS OF POLICE OFFICERS STANDING NEAR VEHICLES IN NEIGHBOURHOOD WHERE SUSPECTED TRUMP RALLY SHOOTER THOMAS MATTHEW CROOKS LIVED

12. VARIOUS OF NEWS CAMERAS

13. VARIOUS OF PENNSYLVANIA STATE POLICE VEHICLE

14. (SOUNDBITE) (English) RETIRED RESIDENT, JIM ZAWOJSKI, 70, SAYING:

"I was stunned. I mean, where is America going to now? Okay. I was talking to one other reporter, and I said, the country's been never so divided. Okay? Everyone needs to go and take a step back. Relax. If you have differences with a candidate, you do it at the voting booth. You know? You don't do it... you don't take action like this clown did. Okay? If it happened to Biden, I'd be just as stunned, you know, because there is no need for that in our society like this."

15. VARIOUS OF TRUMP SUPPORTERS STANDING ON PICKUP TRUCK AND WAVING FLAGS, TRUMP TOWER SEEN IN BACKGROUND

16. VARIOUS OF PICKUP TRUCK PARKED OUTSIDE TRUMP TOWER WITH SUPPORTERS ON BACK WAVING FLAG

17. FLAG WITH ILLUSTRATION OF TRUMP’S FACE ON IT

18. (SOUNDBITE) (English) CALIFORNIA RESIDENT, ARMIN (NO LAST NAME GIVEN), SAYING:

“You know what happened yesterday was a tragedy for this country. It's unthinkable that, you know, something like this can happen. And, you know, I pray for Trump. You know, he's he's trying to save our country from, you know, the radicals that exist. He wants a better USA for us all. And there's now people are trying to assassinate him, which is an unthinkable thing. And, you know, he doesn't have to do this job. He's 78 years old. He should be relaxing, golfing. But he's risking his life, you know, to save the country.”

19. PEOPLE ENTERING TRUMP TOWER

20. ENTRANCE OF TRUMP TOWER

21. TRUMP SUPPORTER HOLDING BANNER, READING (English): 'PURSUED, PERSECUTED, PROSECUTED'

22. STREET NEAR TRUMP’S GOLF CLUB PROPERTY IN BEDMINSTER, NEW JERSEY

23. CAMERA EQUIPMENT / REPORTERS AND VEHICLES SEEN FROM ACROSS THE STREET

24. STREET SIGN READING (English) ‘CLUCAS BROOK RD’

25. SIGNS READING (English) ‘SOMERSET 523 COUNTY’ AND ‘MILE 29’

26. (SOUNDBITE) (English) UNIVERSITY OF CHICAGO PROFESSOR IN THE POLITICAL SCIENCE DEPARTMENT AND THE CHICAGO PROJECT ON SECURITY AND THREATS DIRECTOR, ROBERT PAPE, SAYING:

"Support for political violence matters because that is what normalizes political violence for volatile actors who would go that next step. So it's often the case that the actors who are the lone wolves are, I call them, 'volatile' because they can be mentally ill. They can be inundated with propaganda coming from one of the political parties and so forth. And then and then the media will say, 'Oh, well, that's not really political violence or something.' No, these are not competing explanations."

27. (SOUNDBITE) (English) U.S. PRESIDENT, JOE BIDEN, SAYING:

“Last night, I spoke with Donald Trump. I'm sincerely grateful that he's doing well and recovering."

28. (SOUNDBITE) (English) U.S. PRESIDENT, JOE BIDEN, SAYING:

"Mr. Trump, as a former president, the nominee of the Republican Party already receives a heightened level of security. And I've been consistent in my direction to the Secret Service to provide him with every resource, capability, and protective measure necessary to ensure his continued safety."

29. SCREENSHOT FROM TRUMP'S SOCIAL MEDIA ACCOUNT 'TRUTH SOCIAL' READING (English):

"Thank you to everyone for your thoughts and prayers yesterday, as it was God alone who prevented the unthinkable from happening. We will FEAR NOT, but instead remain resilient in our Faith and Defiant in the face of Wickedness. Our love goes out to the other victims and their families. We pray for the recovery of those who were wounded, and hold in our hearts the memory of the citizen who was so horribly killed. In this moment, it is more important than ever that we stand United, and show our True Character as Americans, remaining Strong and Determined, and not allowing Evil to Win. I truly love our Country, and love you all, and look forward to speaking to our Great Nation this week from Wisconsin. DJT"
</shotlist>

<storyline>
Eyewitness footage at Donald Trump's campaign rally on Saturday (July 13) showed people shouting and panicking during a shooting attack that left the presidential candidate's face streaked with blood. Reuters corroborated visuals from the scene such as the grey structure with glass windows also seen in the video by comparing it to matching file imagery. The date of the video matched that of Reuters and other media reports.

A 2020 high school yearbook photo of Thomas Matthew Crooks of Bethel Park, Pennsylvania, was released on Sunday (July 14) after the FBI identified the 20-year-old as the suspect involved in the attempted assassination of former U.S. President Donald Trump at a campaign rally.

An eyewitness recalled on Sunday (July 14) seeing a man with a riffle on a nearby roof moments before former US president Donald Trump was shot in the ear at a campaign rally in Pennsylvania. Greg Smith said he had alerted police and Secret Services just before shots were heard.

Witnesses who attended the former U.S. President Donald Trump’s rally in Butler, Pennsylvania described the attempted assassination on Sunday (July 14).

Police barriers were set up on Sunday (14 July) near the site of the Donald Trump Pennsylvania rally where a gunman took shots at the Republican presidential hopeful.

A local district attorney in Butler, Pennsylvania, expressed embarrassment on Sunday (July 14) following the assassination attempt on former U.S. President Donald Trump the day before.

Thomas Crooks, the man named by the FBI as the shooter in the attempted assassination of Donald Trump, was "bullied every day" in high school, a high school classmate told media on Sunday (July 14). Jason Kohler, who graduated from Bethel Park High School the year before Crooks, described him as an "outcast" who was "always alone". Kohler said he did not have any interactions with Crooks but spoke of him sitting alone at lunch and wearing hunting outfits to school. The 21-year-old local resident also said he thought Crooks had been on the school's rifle team, though he was uncertain.

The neighbourhood of Bethel Park, Pennsylvania, was abuzz with media and police on Sunday (July 14) a day after one of its residents allegedly tried to kill former U.S. President Donald Trump at a campaign rally about one hour away in the city of Butler.

Residents of the Pittsburgh suburb of Bethel Park expressed shock and sadness on Sunday (July 14) after finding out a local man had been identified as the shooter in an attempted assassination of Donald Trump.

Supporters of former U.S. President Donald Trump gathered outside Trump Tower in New York on Sunday (July 14), after he survived an assassination attempt days before he was due to accept the formal Republican presidential nomination.

It was a quiet scene near the former U.S. President Donald Trump’s property in Bedminster, New Jersey on Sunday (July 14).

In a country already on edge, the assassination attempt on former President Donald Trump has enraged his supporters, paused the Democratic campaign, and raised fears of further political violence in the run-up to November's election. According to analysts, the country is at a tipping point.

U.S. President Joe Biden delivered remarks on Sunday (July 14), one day following the assassination attempt on former President Donald Trump.

Former U.S. President Donald Trump thanked well-wishers on Sunday (July 14) on his social media website Truth Social, after he was shot in the ear in an attempted assassination during a campaign rally.
</storyline>
</example_response>

Here is the input:
<script>
{SCRIPT}
</script>

Please parse the script and push the storyline in <storyline></storyline> tags and the shotlist in <shotlist></shotlist> tags. Both of these should be in <response></response> tags.
Remember to copy both from the script. Don't make any changes to the text.
"""
)

spell_check_chain = (spell_check_prompt | sonnet35).with_config({"run_name": "spell_check"})
get_sot_chain = (get_sot_prompt | sonnet35).with_config({"run_name": "get_sots"})
facts_chain = (facts_prompt | sonnet35).with_config({"run_name": "generate_facts"})
parse_sot_chain = (parse_sot_prompt | sonnet35).with_config({"run_name": "parse_sots"})
reformat_title_chain = (reformat_title_prompt | sonnet35).with_config({"run_name": "reformat_title"})
reformat_chain = (reformat_prompt | sonnet35).with_config({"run_name": "reformat_script"})
sot_chain = (sot_prompt | sonnet35).with_config({"run_name": "add_sots"})
edit_chain = (edit_prompt | sonnet35).with_config({"run_name": "edit_script"})
parse_chain = (parse_prompt | sonnet35).with_config({"run_name": "parse_script"})
logline_chain = (logline_prompt | sonnet35).with_config({"run_name": "logline"})
headline_chain = (headline_prompt | sonnet35).with_config({"run_name": "headline"})
broll_request_chain = (broll_request_prompt | sonnet35).with_config({"run_name": "broll_request"})
broll_chain = (broll_prompt | sonnet35).with_config({"run_name": "broll"})
parse_broll_chain = (parse_broll_prompt | sonnet35).with_config({"run_name": "parse_broll"})
fix_broll_chain = (fix_broll_prompt | sonnet35).with_config({"run_name": "fix_broll"})
match_sot_chain = (match_sot_prompt | sonnet35).with_config({"run_name": "match_sot"})
match_hard_sot_chain = (match_hard_sot_prompt | sonnet35).with_config({"run_name": "match_hard_sot"})
language_to_iso_chain = (language_to_iso_prompt | sonnet35).with_config({"run_name": "language_to_iso"})
match_clip_to_sots_chain = (match_clip_to_sots_prompt | sonnet35).with_config({"run_name": "match_clip_to_sots"})
json_chain = (json_prompt | sonnet35).with_config({"run_name": "json"})
courtesy_chain = (courtesy_prompt | sonnet35).with_config({"run_name": "courtesy"})
extract_storyline_and_shotlist_chain = (extract_storyline_and_shotlist_prompt | sonnet35).with_config({"run_name": "extract_storyline_and_shotlist"})

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
            if type(response_xml['response']) is str:
                return response_xml['response'].strip()
            else:
                return response_xml['response']
        except OperationalError:
            response_raw = chain.invoke(params).content
            response_xml = extract_xml(response_raw)
            if type(response_xml['response']) is str:
                return response_xml['response'].strip()
            else:
                return response_xml['response']
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
