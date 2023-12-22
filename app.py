import streamlit as st
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
import base64
from langchain.schema import AIMessage, HumanMessage

st.title("Multimodal Chat")

llm = Ollama(model="bakllava")


def bind_and_run_llm(payload):
    image = payload["image"]
    prompt = payload["prompt"]
    bound = llm.bind(images=[image])
    return bound.invoke(prompt)


image_template = "{image}"
image_prompt = PromptTemplate.from_template(image_template)
prompt_template = "{question}"
prompt = PromptTemplate.from_template(prompt_template)

chain = (
    {"image": itemgetter("image"), "prompt": prompt} |
    RunnableLambda(bind_and_run_llm)
)


if "messages" not in st.session_state:
    st.session_state["messages"] = [
            (AIMessage(content="How can I help you?"), False)
            ]

if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

for msg in st.session_state.messages:
    if not msg[1]:
        st.chat_message(msg[0].type).write(msg[0].content)
    else:
        st.chat_message(msg[0].type).image(msg[0].data, width=200)

if uploaded_file := st.sidebar.file_uploader("Upload an image file",
                                             type=["jpg", "png"]):
    if st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.messages.append(
                (HumanMessage(
                    content=uploaded_file.name,
                    data=uploaded_file
                    ), True))
        st.chat_message("user").image(uploaded_file, width=200)

if prompt := st.chat_input():
    st.session_state.messages.append((HumanMessage(content=prompt), False))
    st.chat_message("human").write(prompt)

    response = ""
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        b64 = base64.b64encode(data).decode()

        response = chain.invoke({"question": prompt, "image": b64})
    else:
        response = "Please upload an image first"

    st.session_state.messages.append((AIMessage(content=response), False))
    st.chat_message("assistant").write(response)
