import boto3
import json
from dotenv import load_dotenv
import os
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import SpacyTextSplitter


def fileChunker(uploaded_file):
    llm = Bedrock(
        credentials_profile_name="bedrock",
        model_id="anthropic.claude-v2",
        endpoint_url="https://bedrock.us-east-1.amazonaws.com",
        region_name="us-east-1",
        verbose=True
    )
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=3000, chunk_overlap=100
    )

    doc = text_splitter.split_documents(documents)

    # avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents]) // len(documents)
    # avg_char_count_pre = avg_doc_length(documents)
    # avg_char_count_post = avg_doc_length(doc)
    # print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
    # print(f'After the split we have {len(doc)} documents more than the original {len(documents)}.')
    # print(f'Average length among {len(doc)} documents (after split) is {avg_char_count_post} characters.')

    map_custom_prompt = '''

    Human: You are a business professional and you need to create a 500 word summary on the documents provided with no postamble:
    Documents:`{text}`


    Assistant:

    '''

    map_prompt_template = PromptTemplate(
        input_variables=['text'],
        template=map_custom_prompt
    )

    summary_chain = load_summarize_chain(llm=llm,
                                         chain_type="map_reduce",
                                         map_prompt=map_prompt_template,
                                         verbose=True,
                                         token_max=12000)
    # summary_chain = load_summarize_chain(llm=llm, chain_type="refine", refine_prompt=map_prompt_template,
    #                                      verbose=True)

    output = summary_chain.run(doc)
    return output
