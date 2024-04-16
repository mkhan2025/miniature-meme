import os
from apikey import apikey 
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# sets the environment to the apikey
os.environ['OPENAI_API_KEY'] = apikey

# App Framework 
st.title('Write me a story title about')
prompt = st.text_input('Enter story prompt')

# Prompt template --> "write me a story title about " + topic. It takes an input variable and formats the string using that var 
title_template = PromptTemplate(
    input_variables = ['topic'], 
    # {topic} indicates within a string that this is a var 
    template = 'Write me a story title about {topic}'
)
# Prompt template 2
essay_template = PromptTemplate(
    input_variables = ['title','wikipedia_research'], 
    template = 'Generate me a narrative-style story about the title TITLE: {title}  while leveraging this wikipedia research : {wikipedia_research}'
)
# Memory
title_memory = ConversationBufferMemory(input_key = 'topic', memory_key ='chat_history')
essay_memory = ConversationBufferMemory(input_key = 'title', memory_key ='chat_history')

# llms --> for how creative or random your gpt is 
# langchain library to make a OpenAI client to initiate correspondence with OPENAI API, automatically does GET request
# Student(int gpa, boolean canGraduate)
# Student(int gpa = 4.0, boolean canGraduate)
# Student(4.0, True)
# creating an instance of OpenAI class 
llm = OpenAI(temperature = 0.9)
title_chain = LLMChain(llm = llm, prompt = title_template, verbose=True, output_key = 'title', memory=title_memory)
essay_chain = LLMChain(llm = llm, prompt = essay_template, verbose=True, output_key = 'essay', memory=essay_memory)
# sequential_chain = SequentialChain(chains = [title_chain, essay_chain], input_variables = ['topic'], output_variables = ['title', 'essay'], verbose = True)
#library
wiki = WikipediaAPIWrapper()
# Show stuff to the screen
if prompt: 
    # response = sequential_chain({'topic': prompt})
    #API calls 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    essay = essay_chain.run(title=title, wikipedia_research=wiki_research)
    #showing on web page 
    st.write(title)
    st.write(essay)

    with st.expander('Title History'): 
        st.info(title_memory.buffer)
    
    with st.expander('Essay History'): 
        st.info(essay_memory.buffer)

    with st.expander('Wikipedia Research History'): 
        st.info(wiki_research)
