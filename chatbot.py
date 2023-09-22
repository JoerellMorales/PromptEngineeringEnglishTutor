from dotenv import load_dotenv
import guidance

load_dotenv()

guidance.llm = guidance.llms.OpenAI(model="gpt-3.5-turbo")


class GuidancePrompt():
   
    base_prompt = '''
        {{#system~}}
        Your task is to be help me improve my English proficiency by conversing to me about my interests or you try to initiate conversation that is outside my interest, while teaching me about the English language.
        You must strictly follow the grammatical rules while having a conversation with me. 
        To make our discussion more conversational ask me one question at a time and make sure to end your sentence with a question to have a back-and-forth conversation between you and me.
        You must also correct my grammatical errors if there is any and praise me if there is no errors.
        {{~/system}}

        {{#user~}}
        Your task is to be help me improve my English proficiency by conversing to me about my interests or you try to initiate conversation that is outside my interest, while teaching me about the English language.
        You must strictly follow the grammatical rules while having a conversation with me. 
        To make our discussion more conversational ask me one question at a time and make sure to end your sentence with a question to have a back-and-forth conversation between you and me.
        You must also correct my grammatical errors if there is any and praise me if there is no errors.

        Some of the example of grammatical errors are delimeted with triple backticks:
        ```
        - "She not went to the market."
        - "They don't likes pizza."
        - "She plays soccer good."
        - "Me and him are best friends."
        - "The book is more interestinger than the movie."
        - "He runned really fast."
        - "She sings gooder than him."
        - "I'm not hungry, so I don't needs dinner."
        - "We was going to the beach tomorrow."
        - "They was at the concert last night."
        ```
        For each of your response you must reply in english with the corresponding user's native language as well.
        Some of the examples are delimited with triple backticks:
        ```
        1) When language is Korean:'Hello, I am your AI chatbot. How can I help you today? 안녕하세요, 인공지능 챗봇입니다. 오늘은 무엇을 도와드릴까요?'
        2) When language is Filipino: 'Hello, I am your AI chatbot. How can I help you today? Kamusta, ako ang iyong AI chatbot. Paano kita matutulungan ngayon?'
        3) When language is German: 'Hello, I am your AI chatbot. How can I help you today? Hallo, ich bin Ihr KI-Chatbot. Wie kann ich Ihnen heute helfen?'
        ```

        Finally, always be on alert on correcting the grammar of my input if there is a grammatical errors on my input follow this format that is delimited with backticks.
        ```
        English: "<Response in English Text>", Translated: "<Response in user's native langage Text>"
        {{~/user}} 

        {{#assistant~}}
        Ok got this
        {{~/assistant}}

        '''
      


    def __init__(self,role : str,content :str):
        self.role = role
        self.content = content

    def return_model_response(self):
        concate= ''
        
        if self.role == 'User':
            concate = '''       
        {{#user~}}
        {{query}}
        {{~/user}}

        {{#assistant~}}
        {{gen 'reply'}}
        {{~/assistant}}
            '''
        elif self.role == 'Assistant':
            concate = '''
        {{#assistant~}}
        {{query}}
        {{~/assistant}}
            '''
        final_prompt_string = self.base_prompt + f"{concate}"

        model_response = guidance(final_prompt_string)
        prompt = self.content
        response = model_response(query=prompt)
        return  response["reply"]
        
