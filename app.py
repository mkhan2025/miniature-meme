import os
from apikey import apikey 
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

# App Framework 
st.title('Write me a story about')
prompt = st.text_input('Enter story topic')

# Prompt template title
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template = 'Write me a story about {topic}'
)
# Prompt template 2
essay_template = PromptTemplate(
    input_variables = ['title','wikipedia research'], 
    template = 'Generate me a narrative-style story about the title TITLE: {title}  while leveraging this wikipedia research : {wikipedia_research}'
)
# Memory
title_memory = ConversationBufferMemory(input_key = 'topic', memory_key = 'chat history')
essay_memory = ConversationBufferMemory(input_key = 'title', memory_key = 'chat history')

# llms --> for how creative or random your gpt is 
llm = OpenAI(temperature = 0.9)
title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key = 'title', memory=title_memory)
essay_chain = LLMChain(llm = llm, prompt = essay_template, verbose = True, output_key = 'essay', memory=essay_memory)
# sequential_chain = SequentialChain(chains = [title_chain, essay_chain], input_variables = ['topic'], output_variables = ['title', 'essay'], verbose = True)
wiki = WikipediaAPIWrapper()
# Show stuff to the screen
if prompt: 
    # response = sequential_chain({'topic': prompt})
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    essay = essay_chain.run(title=title, wikipedia_research=wiki_research)
    st.write(title)
    st.write(essay)

    with st.expander('Title History'): 
        st.info(title_memory.buffer)
    
    with st.expander('Essay History'): 
        st.info(essay_memory.buffer)

    with st.expander('Wikipedia Research History'): 
        st.info(wiki_research)