from dotenv import load_dotenv
import json
from pydantic import BaseModel, Field
from typing import List,Dict
import guardrails as gd
import guidance
import openai
load_dotenv()

guidance.llm = guidance.llms.OpenAI(model="gpt-4")

#For First Endpoint
class GuidancePrompt():
   
    base_prompt = '''
        {{#system~}}
        I want you to act as a spoken and multilingual AI English teacher. 
        {{~/system}}

        {{#user~}}
        I want you to help me improve my usage of the english language. Check my use of grammar for each of my messages every single time
        Always remember the following information about me:
        My name is {{name}}, My level of proficiency in english is at a {{proficiency}} level and my native language is {{language}}
        Your response should always be in a JSON format specified below within the triple backticks
        The JSON response should contain either your praise or corrections depnding on the use of grammar for the part of the message I specifally asked you to analyzed

        {{~! Analyze my message for any and all grammatical errors ~}}
        
        Make sure that the message also contains correct punctuation, capitalization and other rules of grammar.
        If there are any gramatical errors you should respond immediately with the correct grammar only,
        If there are none you should praise me and continue the conversation with a question to keep the conversation alive
        And of course make sure to add a question to keep the conversation active

        Lastly format your response so that it is followed by a direct translation to my native language of {{language}}
        Be alert always on making corrections from my input and on responding to me with a JSON format delimited with triple backticks: 
        ```\{\{"content": "<Response in English Text>", "translation": "<Response Translated to {{language}} Text>"\}\}```
        
        Start with my next message "Hello, {{name}}! Im Lex, your English AI Tutor. I'm here to help you learn English easily and fun. For starter/ To get started, what topic would you like to talk about?"
        Remember always that You should always check my grammar for each and everyone of my message and to praise/correct me accordingly.

        After your initial JSON response look back at the english part of your response and
        then from that response take 4 words to structure in the JSON format as structured below.
        response and use it in another JSON format as structured below
        
        Your response should always be in a JSON format structured like this:

        \[
            "Word1": \{
                "context-usage": "<insert statement which articulate the specific context usage of the word>",
                "definition": "<insert definition of the word>",
                "1st-example": "<Provide specific beginner use-case example of the word>",
                "2nd-example": "<Provide specific {{proficiency}} use-case example of the word>",
                "3rd-example": "<Provide specific {{proficiency}} use-case example of the word>"
            \},
            "Word2": \{
                "context-usage": "<insert statement which articulate the specific context usage of the word>",
                "definition": "<insert definition of the word>",
                "1st-example": "<Provide specific beginner use-case example of the word>",
                "2nd-example": "<Provide specific {{proficiency}} use-case example of the word>",
                "3rd-example": "<Provide specific {{proficiency}} use-case example of the word>"
            \},
        \]
        {{~/user}}

        {{~! Iterate over each message in the message_history recieved from the [ET_APP_REQUEST] ~}}
        {{~#each message_history}}
        {{#if this.sender == "user"}}
        {{#user~}}{{this.content}}{{~/user}}
        {{else}}
        {{#assistant~}}{{this.content}}{{~/assistant}}                 
        {{/if}}
        {{~/each}}

        {{~! Now that the model is provided with the whole context of conversation, it is now time to provided its response ~}}                  
        {{#assistant~}}
        {{gen 'response'}}
        {{~/assistant}}
        
        '''
    model_response = guidance(base_prompt)
    
    def __init__(self, message_history:List[dict],name : str , language :str , proficiency : str):
        self.message_history = message_history

    def return_model_response(self):
        prompt= self.message_history
        model_response = self.model_response(message_history=prompt,name="Joe",proficiency="Beginner",language="Filipino")
        return model_response['response']


#For Translation
class Generated(BaseModel):
    message: str = Field(description="Simply copy the message here")
    translated: str =Field(description="Translate the message into the user's native language")
class Assistant(BaseModel):
    translation: List[Generated]

#For Vocab
#Note sobrang tagal pag kasama examples nag titimeout siya. If context_usage and definition lang gumagana.
#Di ko magawa yung format as is, In guardrails bawal mag generate ng key yung lamang lang pwede.
class Words(BaseModel):
    word : str = Field(description="Chosen Word")
    context_usage : str = Field(description="insert statement which articulate the specific context usage of the word")
    definition : str = Field(description="insert definition of the word")
    first_example : str = Field(description="Provide specific beginner use-case example of the word")
    second_example : str = Field(description="Provide specific use-case example of the word based off user's proficiency")
    third_example : str = Field(description="Provide specific use-case example of the word based off user's proficiency")
class Vocab(BaseModel):
    sender :str = Field(description="Copy the Sender here")
    vocab: List[Words]



#For 2nd and third endpoint.
class GuardRail():

    translate_prompt ='''
    You are a AI translator that will help the user to enhance his/her ${proficeincy} english proficiency. \
    You will translate the given message into ${language}, keep in mind ${language} is the user's native language.

    ${message}

    ${gr.xml_prefix_prompt}

    ${output_schema}
    
    '''


    vocab_prompt ='''
    You are a AI english tutor that is tasked to increase the user's english proficiency. \
    And I need you to take 4 words from the message that is delimeted by triple backtics.\
    And explain the words so that the user can understand it further to increase english proficiency. \
    Remember the sender is : ${sender}. \
    ```
    ${message}
    ```

    ${gr.xml_prefix_prompt}

    ${output_schema}
    
    '''
    def __init__(self, message: str,proficiency:str, language:str = None, sender: str = None):
        self.message = message
        self.language = language
        self.proficiency = proficiency
        self.sender = sender

    #Translation
    def translate(self):
        response = self.guardrail_translate(prompt=self.message,
                                  language=self.language,
                                  proficiency=self.proficiency,
                                  template=self.translate_prompt,
                                  output=Assistant)
        return response
    
    def guardrail_translate(self,prompt,language,proficiency,template,output):
        guard = gd.Guard.from_pydantic(output_class=output, prompt=template)
        raw_llm_output, validated_output = guard(
            openai.ChatCompletion.create,
            prompt_params={"message":prompt,"language":language,"proficiency":proficiency},
            model="gpt-4-0613",
            max_tokens=2048,
            temperature=0,
        )
        return validated_output
    
    #Vocab
    def vocab(self):
        response = self.guardrail_vocab(prompt=self.message,
                                  proficiency=self.proficiency,
                                  sender=self.sender,
                                  template=self.vocab_prompt,
                                  output=Vocab)
        return response
    
    def guardrail_vocab(self,prompt,sender,proficiency,template,output):
        guard = gd.Guard.from_pydantic(output_class=output, prompt=template)
        raw_llm_output, validated_output = guard(
            openai.ChatCompletion.create,
            prompt_params={"message":prompt,"sender":sender,"proficiency":proficiency},
            model="gpt-4-0613",
            max_tokens=2048,
            temperature=0,
        )
        return validated_output

        
