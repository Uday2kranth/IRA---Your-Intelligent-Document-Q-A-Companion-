import hashlib
import streamlit as st
import sqlite3
import os
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# st.set_page_config(page_title= "your  personal digit nurse", page_icon=":robot_face:")
# st.title("healthMate AI")
# st.subheader("your digital nurse at your service")
# st.write("welcome to healthMate AI, your personal digital nurse designed to assist you with all your healthcare needs. Whether you have questions about symptoms, medications, or general health advice, healthMate AI is here to help.")

load_dotenv()
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

gemini = genai.GenerativeModel("gemini-2.0-flash")

DB_NAME = "IRA.db"
UPLOAD_DIR = "uploads"

def init_db():
     with sqlite3.connect(DB_NAME) as conn:
         conn.execute("""
                      CREATE TABLE IF NOT EXISTS users (
                          user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                          first_name  TEXT NOT NULL,
                          last_name TEXT NOT NULL,
                          date_of_birth TEXT NOT NULL,
                          email text NOT NULL UNIQUE,
                          password TEXT NOT NULL
            )
            """)
         
         conn.execute("""
                      CREATE TABLE IF NOT EXISTS files (
                          file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                          user_id INTEGER NOT NULL,
                          file_name TEXT NOT NULL,
                          file_path TEXT NOT NULL,
                          FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            """)

     print("Database and tables both are initialized.")
     
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(first_name, last_name, date_of_birth, email, password):
    with sqlite3.connect(DB_NAME) as conn:
        try:
            conn.execute("""
                         INSERT INTO users(first_name,last_name,date_of_birth,email,password)
                         VALUES(?,?,?,?,?)
                         """, (first_name,last_name,date_of_birth,email,hash_password(password)))
            conn.commit()
            return True,"Account created successfully, Now you can login"
        except sqlite3.IntegrityError:
            return False,"this email is already registered, try logging in."
            
            
def login(email,password):
    with sqlite3.connect(DB_NAME) as conn:
        user = conn.execute("""
                            SELECT user_id,first_name,last_name FROM users WHERE email=? AND password=?
                            """, (email,hash_password(password))).fetchone()
        return user if user else None

def save_file(user_id,file_name,file_path):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
        INSERT INTO files(user_id,file_name,file_path)
        VALUES(?,?,?)
        """, (user_id,file_name,file_path))
        conn.commit()
        
def get_user_files(user_id):
    with sqlite3.connect(DB_NAME) as conn:
        files= conn.execute("""
        SELECT file_name, file_path FROM files WHERE user_id=?
        """,(user_id,)).fetchall()
        return files
    
def delete_file(user_id,file_name):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
        DELETE FROM files WHERE user_id=? AND file_name=?
        """, (user_id,file_name))
        conn.commit()
init_db()


# # Initialize embeddings model
# from sentence_transformers import SentenceTransformer
# embeddings = SentenceTransformer(model_name="all-MiniLM-L6-v2")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_chunks(text):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    db = FAISS.from_texts(chunks, embeddings)
    return db
def get_rel_text(user_query,db):
    rel_text =  db.similarity_search(user_query,k=1)
    return rel_text[0].page_content if rel_text else "no relevant information found"

def bot_response(model, query, relevant_texts, history,):
    context= "".join(relevant_texts)
    prompt = f"""
    
    This is the context of the document
    context: {context}
    and this is the user query
    user : {query}
    And this is the history of the conversation
    History : {history}
    
    Please generate a response to the user query based on the context and the history of the conversation.
    The questions might be asked related to the provided context, and may also be in terms of any topics and  mostly it will be in the context of the uploaded document.
    Answer the question with respect to the context provided, you can also use your additional knowledge too, but do not make up facts.
    Answer the following queries like a professional could answer them , having a lot of knowledge on the basis of document's context.
    
    Answer in a very friendly and interactive way, do not make it sound like a robot.
    Bot :

    
    """
    
    response = model.generate_content(
        prompt,
        generation_config = genai.GenerationConfig(
            temperature = 0.68
        )
    )

    return response.text

model = genai.GenerativeModel(model_name="gemini-2.0-flash",
    system_instruction="""
    your name is ira and q and A plus questions answering gives answer to the user query regarding the topics or text  in the files uploaded   or  ellaborate words or sentences or paragraphs in the files uploaded by the user based on the context of the file uploaded by the user and also answer the questions asked by the user. also you need to use  your own knowledge to answer the user query when user ask it you to do so
    your role

    your Roles:
    1) your are a Q and A bot, who is intelligent in finding the particular answers based on the context provided and also you need to use  your own knowledge to answer the user query when user ask it you to do so
    2) you are very intelligent at making connections between the context provided and the user query and also you  have ability to find the text and context and content in the uploaded files and you very good at it 
    3) you never hallucinate and you always give the correct answer to the user query based on the context provided in the uploaded files

    Points to remember:
    1) You should engage with the user like a fellow scholar or peer or greatest mind without arrogance, and give the user proper reply for his queries
    2) The concentration and the gist of the conversation no need to be completely based on the content  in files as long you give right answer and correct the content in the uploaded files, your flow of chat should be more like a human conversation.
    3) If the conversation goes way too out of context (out of context = others topics which are not in the uploaded files) or if the user input is abusive, let the user know that the content is abusive and we cannot tolerate such inputs.
    4) The important part is that you should not anywhere mention user to, "read the document or file himself or ask him for help to assist you"
    5) The highest priority should be , and you if you think content in the file wrong you should mention that to user
    
    """)

st.set_page_config(page_title="IRA - Your Intelligent Document Q&A Companion", page_icon="👩‍💻" , layout="wide")

if 'messages' not in st.session_state:
    st.session_state.messages={}

with st.sidebar:
    selected = option_menu(
        "Menu",["Landing Page","Login/Signup","Upload Documents dude","Chat With IRA", "Document Manager"],
        icons = ["house","Person-plus","Cloud-upload","Chat-dots-fill","files"],
        menu_icon = "mortarboard",default_index=0
    )

if selected == "Login/Signup":
    st.header("Login/Signup")
    
    if "user_id" in st.session_state:
        st.info(f"You are Logged in as {st.session_state['first_name']} {st.session_state['last_name']}")
        if st.button("LogOut"):
            st.session_state.clear()
            st.success("Logged OPut Successfully!")
            
    else:
        action = st.selectbox("select an action",['Login','Signup'])
        
        if action == "Signup":
            st.subheader("sign up")
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            dob =st.date_input("Date of Birth genius!")
            email = st.text_input("Email")
            password = st.text_input("password",type='password')
            
            if st.button("Sign Up"):
                success,msg = sign_up(first_name,last_name,dob,email,password)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
                    
        elif action == "Login":
            st.subheader("Login")
            email = st.text_input("Email")
            password=st.text_input("password",type='password')
            
            if st.button("Login"):
                user = login(email,password)
                if user:
                    st.session_state['user_id'],st.session_state['first_name'],st.session_state['last_name']=user
                    st.success(f"Looged in as : {user[1]} {user[2]}!")
                    st.session_state.messages[st.session_state['user_id']]=[]
                else:
                    st.error("invalid Email or Password , Please check your login credentials first and try again")
                    
# st.write(st.session_state)
if selected == "Chat With IRA":
    st.header("chat with IRA")
    if 'user_id' not in st.session_state:
        st.warning("Please login to chat with IRA")
        
    else:
        st.info(f"welcome {st.session_state['first_name']}")
        st.write("Please go ahead and provide the context you want to chat about ")
        chat_history = st.session_state.messages.get(st.session_state['user_id'],[])
        
        history = []
        for msg in chat_history:
            if msg['role'] == 'user':
                user_msg = {
                    'role':'user',
                    'parts':msg['content']
                }
                history.append(user_msg)
            else:
                bot_msg = {
                    'role': 'model',
                    'parts':msg['content']
                }
                history.append(bot_msg)
                
        #start chat session and chats in all this session will be stored in the history variable and can be remembered by llm 
        chat_bot = model.start_chat(history =history)
        #display previous chat messages in the same chat sessions 
        for message in chat_history:
            if message['role'] == 'user':
                st.chat_message(message['role']).markdown(message['content'])
            else:
                st.chat_message(message['role']).markdown(message['content'])
                
        user_question = st.chat_input("input your message here:")
        
        if user_question:
            #show user messages in the session as well 
            st.chat_message("user").markdown(user_question)
            #add to chat history 
            chat_history.append({
                'role' : 'user',
                'content' : user_question
            })
            #code for genetring bot reponse 
            with st.spinner("let me thining>>>>!"):
                response =chat_bot.send_message(user_question)
                
                #show the bot responses in the session 
                st.chat_message("assistant").markdown(response.text)
                #adding bot repsonse to history 
                chat_history.append({
                    'role':'assistant',
                    'content' :response.text
                })
                #save updated chat history in the session state
            st.session_state.messages[st.session_state['user_id']]=chat_history
# upload the documents details 
if selected == "Upload Documents dude":
    st.subheader("Upload Documents dude")
    
    if 'user_id' not in st.session_state:
        st.warning("please login to access the Medical record bot")
    else:
        # feature selection with two options upload the document ,  and chat with ira regarding the document uploaded
        with st.expander("select the feature",expanded =True):
            choice = st.radio(
                label="Select of the below options",
                options =["upload Document","ask IRA queries  about document "]
            )
        st.info(f"Welcome{st.session_state['first_name']}! you can upload your documents here and IRa will help you in finding the answers to your queries regarding the document uploaded")
        
        #file upload section
        if choice == "upload Document":
            #file uploader icon
            file=st.file_uploader(label="upload your document here",type=['pdf'])
            
            if file:
                file_name=file.name
                #create unique file paths for each user
                file_path = os.path.join(UPLOAD_DIR,f"{st.session_state['user_id']}_{file_name}")
                
                #make sure upload directory exists
                os.makedirs(UPLOAD_DIR,exist_ok=True)
                #save fie button to save file once it is uploaded 
                if st.button("save_file"):
                    #save the file content  what cause issue in the sir's code
                    with open(file_path,'wb') as f:
                        f.write(file.getbuffer())
                    #then save file info to data base 
                    save_file(st.session_state['user_id'],file_name,file_path)
                    st.success(f"File{file_name} saved successfully")
                    
            #show uploaded files 
            st.subheader("your uploaded files")
            files = get_user_files(st.session_state['user_id'])
            
            if files:
                #list all files with delete option
                for file_name, file_path in files:
                    st.markdown(f"**{file_name}**")
                    if st.button(f"delete {file_name}"):
                        delete_file(st.session_state['user_id'],file_name)
                        #also remove physicsl file 
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        st.success(f"file {file_name} deleted successfully")
                #file content viewr part of the code 
                st.subheader("file content viewer")
                s_file = st.selectbox(
                    label= "select a file to view its content",
                    options = [filename for filename, filepath in files]
                )
                #function to get the file paths 
                def get_value(filename,file_list):
                    for name,path in file_list:
                        if name == filename:
                            return path
                    return None
                
                if s_file:
                    files_path = get_value(s_file,files)
                    if st.button("view content"):
                        with st.spinner("analyzing the content in the document"):
                            pdf_reader = PdfReader(file_path)
                            text=''
                            for page in pdf_reader.pages:
                                text += page.extract_text()
                                
                            st.subheader(f"content of the file {s_file}")
                            st.write(text)
            else:
                st.info("No document uploaded yet dude")
        #Chat with medical report bot section (Rag impletations with robust error handling)
        elif choice == "ask IRA queries  about document ":
            files =get_user_files(st.session_state['user_id'])
            
            if not files:
                st.warning("please upload a document to chat with IRA")
            else:
                # Step 1: File selection
                st.subheader("📄 Select Document for IRA to analyze")
                selected_file = st.selectbox(
                    "Choose a medical report:",
                    options=[filename for filename, _ in files]
                )
                
                if selected_file:
                    #step 2: find the file path
                    file_path=None
                    for filename,path in  files:
                        if filename == selected_file:
                            file_path = path
                            break
                    
                    # Step 3: Process PDF and create vector database (only once per file)
                    if f"processed_{selected_file}" not in st.session_state:
                        st.info("🔄 Processing your document... This may take a moment.")
                        
                        # Read PDF content with error handling
                        with st.spinner("Reading PDF..."):
                            try:
                                # Check if file exists and has content
                                if not os.path.exists(file_path):
                                    st.error(f"File {selected_file} not found. Please upload the file again.")
                                    st.stop()
                                
                                if os.path.getsize(file_path) == 0:
                                    st.error(f"File {selected_file} is empty. Please upload a valid PDF file.")
                                    st.stop()
                                
                                pdf_reader = PdfReader(file_path)
                                full_text = ''
                                for page in pdf_reader.pages:
                                    full_text += page.extract_text()
                                
                                if not full_text.strip():
                                    st.error("No text found in the PDF. Please upload a text-based PDF file.")
                                    st.stop()
                                    
                            except Exception as e:
                                st.error(f"Error reading PDF file: {str(e)}")
                                st.error("Please make sure you uploaded a valid PDF file and clicked 'save_file'.")
                                st.stop()
                        
                        # Create vector database using RAG approach
                        with st.spinner("Creating knowledge base..."):
                            # Split text into chunks
                            text_chunks = get_chunks(full_text)
                            
                            # Create vector store for similarity search
                            vector_db = get_vector_store(text_chunks)
                            
                            # Store processed data in session
                            st.session_state[f"processed_{selected_file}"] = {
                                'text': full_text,
                                'vector_db': vector_db
                            }
                        
                        st.success("✅ Document processed successfully dude!")
                    
                    # Step 4: Chat Interface
                    st.subheader(f"💬 Chat about: {selected_file}")
                    st.write("Ask questions about your document!")
                    
                    # Initialize chat history for this specific file
                    chat_key = f"chat_{st.session_state['user_id']}_{selected_file}"
                    if chat_key not in st.session_state:
                        st.session_state[chat_key] = []
                    
                    # Display previous chat messages
                    for message in st.session_state[chat_key]:
                        if message['role'] == 'user':
                            st.chat_message("user").markdown(f"**You:** {message['content']}")
                        else:
                            st.chat_message("assistant").markdown(f"**IRA:** {message['content']}")
                    
                    # Step 5: Handle new user questions
                    user_question = st.chat_input("Ask me anything about your document dude...")
                    
                    if user_question:
                        # Add user question to chat
                        st.session_state[chat_key].append({
                            'role': 'user', 
                            'content': user_question
                        })
                        st.chat_message("user").markdown(f"**You:** {user_question}")
                        
                        # Process question using RAG system
                        with st.spinner("🔍 Analyzing your document..."):
                            # Get processed data
                            processed_data = st.session_state[f"processed_{selected_file}"]
                            vector_db = processed_data['vector_db']
                            
                            # Find relevant information using similarity search
                            relevant_info = get_rel_text(user_question, vector_db)
                            
                            # Get recent chat history for context
                            recent_chat = []
                            for msg in st.session_state[chat_key][-4:]:  # Last 4 messages only
                                recent_chat.append(f"{msg['role']}: {msg['content']}")
                            chat_history_text = "\n".join(recent_chat)
                            
                            # Generate response using retrieved information
                            bot_answer = bot_response(
                                model,
                                user_question, 
                                [relevant_info],  # relevant text from document
                                chat_history_text
                            )
                            
                            # Add bot response to chat
                            st.session_state[chat_key].append({
                                'role': 'assistant', 
                                'content': bot_answer
                            })
                            st.chat_message("assistant").markdown(f"**IRA:** {bot_answer}")

# LANDING PAGE
if selected == "Landing Page":
    st.title("👩‍💻 IRA - Your Intelligent Document Q&A Companion")
    st.markdown("### Your AI-Powered Document Analysis Companion dude")
    
    # Create two columns for features
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💬 Chat with IRA - General Conversation")
        st.write("Chat with IRA about anything dude")
        st.info("Have normal conversations with your AI companion")
        
    with col2:
        st.subheader("📄 Document Analysis Bot")
        st.write("Upload and chat with your documents using AI")
        st.info("Ask questions about your uploaded documents dude")
    
    # Show login status
    if 'user_id' not in st.session_state:
        st.warning("⚠️ Please login to access all features dude!")
    else:
        st.success(f"👋 Welcome back, {st.session_state['first_name']} genius!")

# DOCUMENT MANAGER PAGE
if selected == "Document Manager":
    st.subheader("Document Manager")
    
    if 'user_id' not in st.session_state:
        st.warning("please login to access the Document Manager dude")
    else:
        st.info(f"Welcome{st.session_state['first_name']}! Manage all your documents here")
        
        files = get_user_files(st.session_state['user_id'])
        
        if files:
            st.subheader("Your uploaded documents dude")
            for file_name, file_path in files:
                with st.expander(f"📄 {file_name}"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**File:** {file_name}")
                        st.write(f"**Path:** {file_path}")
                    
                    with col2:
                        if st.button(f"View", key=f"view_{file_name}"):
                            try:
                                pdf_reader = PdfReader(file_path)
                                text = ''
                                for page in pdf_reader.pages:
                                    text += page.extract_text()
                                st.text_area("Content:", text, height=200)
                            except Exception as e:
                                st.error(f"Error reading file: {str(e)}")
                    
                    with col3:
                        if st.button(f"Delete", key=f"delete_{file_name}"):
                            delete_file(st.session_state['user_id'], file_name)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            st.success(f"File {file_name} deleted successfully dude!")
                            st.experimental_rerun()
        else:
            st.info("No documents uploaded yet dude. Go to 'Upload Documents dude' to add some files!")

# st.write(st.session_state)
