import streamlit as st
from pathlib import Path
import os
from doc_summarizer import fileChunker
import time

st.markdown("<h1 style='text-align: center; color: red;'>Long Document Summarization</h1>", unsafe_allow_html=True)


def main():
    if "query_text" not in st.session_state:
        st.session_state["query_text"] = []

    if "query" not in st.session_state:
        st.session_state["query"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    with st.container():
        st.header('Single File Upload')
        File = st.file_uploader('Upload a file', type=["pdf"], key="new")
        user_input = st.session_state["new"]

        if File is not None:
            st.session_state.past.append(user_input)
            save_folder = "/Users/rdoty/Desktop/new_coding_projects/Amazon-Bedrock-Summarization-Long-Document-POC"
            save_path = Path(save_folder, File.name)
            with open(save_path, mode='wb') as w:
                w.write(File.getvalue())

            if save_path.exists():
                st.success(f'File {File.name} is successfully saved!')
                full_path = save_path.absolute()
                my_path = full_path.as_posix()
                start = time.time()
                st.write(fileChunker(my_path))
                end = time.time()
                print(end - start)
                print("DELETE")
                os.remove(save_path)


def clear_session():
    for key in st.session_state.keys():
        del st.session_state[key]


if __name__ == "__main__":
    main()
