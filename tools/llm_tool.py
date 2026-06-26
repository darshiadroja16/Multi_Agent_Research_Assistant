from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv() # read the groq api key 
client = Groq(api_key=os.getenv("GROQ_API_KEY")) #to build connection with groq api 


def ask_llm(user_prompt,system_prompt=None,json_mode=False): # main function to ask llm 
    """
    Groq LLM se answer lo.
    Parameters:
        user_prompt   : actual question ya task
        system_prompt : LLM ka role define karo
        json_mode     : True karo agar JSON output chahiye
    """
    messages =[] # is single API call ke liye system + user messages yahan jaayenge
    if system_prompt:
        messages.append({ # if user has given the role to the system like teacher or tutor then it will store role as the user has given 
            "role":"system",
            "content":system_prompt
        })

    messages.append({ # this is for the user query passing 
        "role":"user",
        "content":user_prompt
    })

    response_format = {"type":"json_object"} if json_mode else None # if true answer will be in json formate else in english

    #this is the actual call to the api

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=messages, # list made above
        temperature = 0.1, # factual answer =0 & creative or varied answer =1
        max_tokens=1000, # 750 words according to 1000 tokens 
        response_format= response_format # format of the response 
    )
    return response.choices[0].message.content # API call given multiple answers so it is given [0] to choose the first one and message.content is used to get the text  


print("1st Test : simple llm call")

answer = ask_llm(user_prompt ="What is self-attention in transformers? Explain in 3 lines.",
            system_prompt="You are a research assistant. Be concise.")
print(answer)


print("2nd test : Grounding")


print("A : Without Grounding")
answer_no_grounding =ask_llm(user_prompt="What exact BLEU score did the 2017 transformer paper achieve?",system_prompt="You are a research assistant.")
print(answer_no_grounding)

print("B : With Grounding")
context = """
Paper: Attention is All You Need (Vaswani et al., 2017)
Result 1: Achieved 28.4 BLEU on WMT 2014 English-to-German translation
Result 2: Achieved 41.0 BLEU on WMT 2014 English-to-French translation
Result 3: Training took 3.5 days on 8 P100 GPUs for big model
Result 4: Base model achieved 27.3 BLEU with significantly less compute
"""
grounded_prompt = f"""Answer ONLY from the context below.
If answer is not in context, say exactly:
'This information is not in the provided documents.'

Context:
{context}

Question: What exact BLEU score did the transformer achieve?"""
answer_grounded =ask_llm(user_prompt=grounded_prompt,
    system_prompt="You are a research assistant.Answer strictly from context only.")

print(answer_grounded)

print("3rd Test : JSON format output")

json_prompt = """Analyze this research topic and respond in JSON format.

Topic: Transformer vs LSTM for NLP tasks

Respond with exactly these keys:
{
  "key_findings": [list of 3 findings],
  "winner": "transformer or lstm",
  "reason": "one line reason"
}"""
import json

json_output = ask_llm(user_prompt=json_prompt,system_prompt="You are a research analyst. Respond only in valid JSON.",json_mode=True)

parsed = json.loads(json_output)
print("Parsed JSON:")
print(f"Key Findings:")
for f in parsed['key_findings']:
    print(f"    → {f}")
print(f"  Winner : {parsed['winner']}")
print(f"  Reason : {parsed['reason']}")
