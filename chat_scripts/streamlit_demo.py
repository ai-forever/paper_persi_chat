from functools import reduce
import json
import pickle
import streamlit as st
from streamlit_chat import message
from streamlit_searchbox import st_searchbox

from scipdf_parser import sci_parse_pdf_article

import string
exclude = set(string.punctuation) - set('-')

from pipeline_streamlit import PersistentChatBot

st.set_page_config(  # Alternate names: setup_page, page, layout
   layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
   initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
   page_title="PaperPersiChat",  # String or None. Strings get appended with "• Streamlit".
   page_icon="memo"
)

@st.cache_data()
def load_titles():
    with open('../papers_segmented_data/titles_all.pkl', 'rb') as f:
        titles = pickle.load(f)
    return titles


def load_paper(paper_title):
    paper_title_prep = paper_title.replace('/', '').replace('"', '_').replace(':', '_')
    with open('../papers_segmented_data/all_papers/' + paper_title_prep + '.json', 'r') as f:
        return json.load(f)
    
    
@st.cache_data()    
def upload_local_paper(pdf_file):
    result = sci_parse_pdf_article(pdf_file)
    return result['title'], result


@st.cache_data()
def load_keywords():
    with open('../papers_segmented_data/words2papers_all.pkl', 'rb') as f:
        words2titles = pickle.load(f)
    return words2titles


def load_pipeline():
    return PersistentChatBot(version='v2', solve_corefs=True, device='cuda:0')


class PersistentChatBotStreamlit:
    def __init__(self):
        self.persistent_chatbot = load_pipeline()

    def show_conversation(self) -> None:
        if not st.session_state.messages:
            ai_content = self.persistent_chatbot.generate_response("")
            st.session_state.messages.append({"role": "assistant", "content": ai_content})
            st.session_state.generated.append(ai_content)
            st.session_state.past.append("")

        if st.session_state.add_button:
            st.session_state.messages.append({"role": "user", "content": st.session_state.user_text})

            ai_content = self.persistent_chatbot.generate_response(st.session_state.user_text)
            st.session_state.messages.append({"role": "assistant", "content": ai_content})

            st.session_state.past.append(st.session_state.user_text)
            st.session_state.generated.append(ai_content)

            st.session_state.add_button = False
            st.session_state.user_text = ""

        if st.session_state.generated:
            for i in range(len(st.session_state.generated)):
                if i > 0:
                    # bot starts
                    user_mess, key = st.session_state.past[i], str(i) + "_user"
                    message(user_mess, is_user=True, key=key, seed=273)
                bot_mess, key = st.session_state.generated[i], str(i)
                message(bot_mess, key=key, seed=87)


def clear_chat() -> None:
    st.session_state.generated = []
    st.session_state.past = []
    st.session_state.messages = []
    st.session_state.user_text = ""
    st.session_state.add_button = False
    st.session_state.paper = None
    st.session_state.paper_init = True

    
def show_chat_buttons() -> None:
    st.sidebar.button(label='Clear chat', on_click=clear_chat)


def press_button():
    st.session_state.add_button = True
    
    
def show_text_input() -> None:
    st.text_area(label="Input your response and press the Enter button below (write a query that is not a question to move to the next segment):", value=st.session_state.user_text, key="user_text")


def get_user_input():
    endings = [
        "We have discussed all sections of the paper. Bye!",
        "I'm sorry that I caused negative emotions :("
    ]
    if len(st.session_state.generated) == 0 or st.session_state.generated[-1] not in endings:
        show_text_input()
        st.button('Enter', on_click=press_button)
    elif len(st.session_state.generated) != 0 or st.session_state.generated[-1] in endings:
        st.text('The dialogue has been completed.')


def show_chat_buttons() -> None:
    st.sidebar.button(label='Clear chat', on_click=clear_chat)
    
def show_chat_buttons2() -> None:
    st.button(label='Clear chat', on_click=clear_chat, key='mainclear')
    
    
def search(query):
    title_sets = [words2titles[word] for word in query.lower().split() if word in words2titles]
    inds = reduce(lambda x, y: x & y, title_sets) if len(title_sets) > 0 else []
    return [titles[i] for i in list(inds)[:100]]


titles = load_titles()
words2titles = load_keywords()


def main():
    st.markdown('<style>div[class="css-6qob1r e1fqkh3o3"] { background: linear-gradient(#FFFBF3,#FDCEC4); } </style>',
                unsafe_allow_html=True)
    st.sidebar.title("PaperPersiChat")
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = PersistentChatBotStreamlit()
    
    select_option = st.sidebar.radio('Paper selection option:', options=['Search in the database', 'Select from the suggested list', 'Upload PDF'], index=0)
    
    with st.sidebar:
        st.markdown("""---""")
        if select_option ==  'Select from the suggested list':
            paper = st.selectbox("Paper", titles[47651:48651])
            meta = None
        elif select_option ==  'Search in the database':
            paper = 'Building Language Models for Text with Named Entities'
            st.markdown('Enter title keywords separated by spaces to find relevant papers')
            paper = st_searchbox(
                lambda x: search(x),
                key="searchbox",
                default = 'Building Language Models for Text with Named Entities',
                clearable=True,
                clear_on_submit=False
            )
            meta = None
        else:
            st.markdown('This method is in a test mode. Try searching the database first. In addition, it is best to load articles from arXiv.')
            paper = 'Building Language Models for Text with Named Entities'
            meta = None
            in_pdf = st.sidebar.file_uploader('Upload your .pdf file with paper text', type="pdf")
            if in_pdf:
                paper, meta = upload_local_paper(in_pdf)
    
    st.sidebar.markdown("""---""")
    
    if 'paper' not in st.session_state or st.session_state['paper'] is None:
        st.session_state.paper = paper
    else:
        if st.session_state.paper != paper:
            clear_chat()
    
    if "generated" not in st.session_state:
        st.session_state.generated = []
    if "past" not in st.session_state:
        st.session_state.past = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_text" not in st.session_state:
        st.session_state.user_text = ""
    if "add_button" not in st.session_state:
        st.session_state.add_button = False
    if "paper_init" not in st.session_state or st.session_state.paper_init:
        st.session_state.paper_init = False
        if meta is None:
            st.session_state.chatbot.persistent_chatbot.init_paper_dialogue(load_paper(paper))
        else:
            st.session_state.chatbot.persistent_chatbot.init_paper_dialogue(meta)
    
    st.markdown("**Title:** " + st.session_state.chatbot.persistent_chatbot.paper_title)
    if len(st.session_state.chatbot.persistent_chatbot.paper_meta) > 0:
        paper_meta = st.session_state.chatbot.persistent_chatbot.paper_meta
        for key in paper_meta:
            st.markdown(f"***{key.title()}:*** " + str(paper_meta[key]))
        st.markdown("""---""")      
        
    segments = []
    for _, segment_meta in st.session_state.chatbot.persistent_chatbot.segmented_paper:
        segments.append(", ".join([repr(el["title"] if str(el['title']) != 'nan' else 'Not parsed title') for el in segment_meta]))  
    choice_segm = st.sidebar.radio('**Segments progress**. If needed, select the option to move and send an empty message to continue the dialogue.', segments, index=max(0, st.session_state.chatbot.persistent_chatbot.cur_segment_num))
    choice_segm = [j for j, s in enumerate(segments) if s == choice_segm][0]
    
    st.sidebar.markdown("""---""")
    
    if not (st.session_state.chatbot.persistent_chatbot.cur_segment_num == -1 and choice_segm == 0) and\
                st.session_state.chatbot.persistent_chatbot.cur_segment_num != choice_segm:
        st.session_state.chatbot.persistent_chatbot.cur_segment_num = int(choice_segm)
        st.session_state.chatbot.persistent_chatbot.dialogue_state = 'block_suggested'
    
    show_chat_buttons()
    
    st.sidebar.markdown('***Hints***')
    st.sidebar.markdown('''
1. The dialogue is constructed step by step using the loop over the paper segments. In the QA session, it is not forbidden to ask questions throughout the entire paper, not just the segment, and the bot will be able to answer them. However, it is advisable to skip sections that you are not interested in.

2. To exit the Q&A mode, write a query that is not a question (e.g., "ok", "thank you", "let's discuss further", etc.).

3. If you want to clear the dialogue history and restart chat, just press the “Clear chat” button in the settings above.
''')
    
    st.session_state.chatbot.show_conversation()
    get_user_input()
    st.markdown("""---""")
    show_chat_buttons2()
    
    
if __name__ == '__main__':
    main()
    