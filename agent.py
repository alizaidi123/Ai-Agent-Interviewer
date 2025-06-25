import streamlit as st
import openai
import json
import os
import io
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv
from audiorecorder import audiorecorder as st_audiorecorder
from pydub.audio_segment import AudioSegment

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in your .env file or system environment.")
    st.stop()

openai.api_key = openai_api_key

# --- Predefined User Credentials and Initial Credits (for demo purposes) ---
# An 'initial_credits' value of -1 means unlimited credits.
PREDEFINED_USERS = {
    "Admin": {"password": "Admin123", "initial_credits": -1},  # Admin has unlimited credits
    "candidate1": {"password": "pass123", "initial_credits": 50},
    "user_test": {"password": "testpass", "initial_credits": 20}
}

INTERVIEW_COST = 5  # Credits deducted per interview


# --- Helper Functions for File Processing ---
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(docx_file):
    document = Document(docx_file)
    text = "\n".join([paragraph.text for paragraph in document.paragraphs])
    return text


def process_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    else:
        st.warning("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
        return None


# --- AI Core Logic Functions ---
def analyze_documents_and_prepare_for_interview(jd_text, resume_text):
    st.info("Analyzing Job Description and Resume to prepare your personalized interview plan...")
    try:
        messages = [
            {"role": "system", "content": f"""You are an AI Interview Analyst, providing feedback to a candidate who just completed a rigorous practice interview.
             You have the full transcript of the interview and the initial interview plan (which includes JD/Resume analysis and identified "relevant_expert_terms").
             Your task is to:
             1. Evaluate the candidate's overall performance against the job description requirements and considering the rigor of the interview questions asked.
             2. Identify the candidate's strong suits based on their answers and how well they handled challenging questions. **Specifically, note instances where the candidate effectively used the identified "relevant_expert_terms" (from the initial plan) to demonstrate expertise.**
             3. Identify the candidate's weak suits or areas for improvement, specifically noting if they struggled with probing questions, provided vague answers, or showed inconsistencies. **Also, highlight opportunities where the candidate *could have used* relevant "expert terms" but did not, or used them incorrectly, impacting their perceived expertise.**
             4. Provide a clear recommendation on whether the candidate is well-prepared for the position (e.g., "Highly Prepared", "Well Prepared", "Needs More Practice", "Significant Improvement Needed").
             5. Give a concise, professional summary of the interview and key takeaways for the candidate's preparation.
             Initial interview plan details for context: {json.dumps(initial_plan, indent=2)}
             Provide the output in a structured JSON format with keys: "overall_qualification",
             "strong_suits", "weak_suits", "recommendation", "interview_summary"."""},
        ]

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"}
        )
        plan = json.loads(response.choices[0].message.content)
        return plan
    except openai.APIError as e:
        st.error(f"OpenAI API error during planning: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Failed to parse JSON response from OpenAI. Please check the API response for errors or try again.")
        st.write(
            f"Raw response (if available): {response.choices[0].message.content if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during planning: {e}")
        return None


def conduct_interview_turn(conversation_history, current_main_question, candidate_response, interview_plan_details):
    st.info("Processing your answer and formulating the next question...")

    # The AI's continuous system prompt with emphasis on rigorous testing and cornering
    system_prompt = f"""You are an extremely rigorous and intelligent HR Interview AI, acting as an interviewer for a candidate preparing for a real interview. Your goal is to thoroughly test the candidate,
    identify strengths, weaknesses, and probe deep into their experience and claims, just like a human interviewer would.

    Based on the full conversation history, the candidate's response to the last question,
    and the original interview plan (which details the JD, Resume analysis, and initial questions), decide on the next action.

    Your actions should be strategic and aim to:
    1.  **Probe Deeply:** Always ask follow-up questions to clarify vague answers, challenge inconsistencies, or delve deeper into experiences mentioned. Demand specific examples, metrics, or detailed steps they took.
    2.  **Corner the Candidate (Rigorous Testing):** If an answer seems superficial, incomplete, or deviates, ask a direct, challenging follow-up. Connect their response directly to the JD's requirements or their own resume claims, highlighting any apparent gaps or overstatements. Do not let vague answers pass; push for concrete details.
    3.  **Assess Problem-Solving & Critical Thinking:** Pose hypothetical scenarios or ask how they would apply their skills to new challenges related to their last answer or the JD.
    4.  **Manage Interview Flow:** Only move to the next *main* question from the pre-generated plan if the current line of questioning is exhausted and you are satisfied with the depth and rigor of the candidate's response on that topic. You are free to ask multiple follow-ups on a single point.
    5.  **Conclude Wisely:** Conclude the interview only when you have gathered sufficient information to make a solid assessment across all key areas required by the JD, or if all core questions and their relevant follow-ups have been thoroughly explored.

    Provide a concise internal assessment of the candidate's response to the *last question*.
    Return your response in JSON format with keys:
    "action" (e.g., "ask_follow_up", "ask_next_question", "conclude_interview"),
    "next_question" (if "action" is ask_follow_up or ask_next_question),
    "internal_assessment_of_last_response",
    "reason_for_action" (explaining why you chose this action, e.g., "candidate response was vague, probing for more detail", "candidate answered comprehensively, moving to next core question").

    Original Interview Plan for Context and Rigor: {json.dumps(interview_plan_details, indent=2)}
    """

    messages = conversation_history + [
        {"role": "user",
         "content": f"AI's last question: {current_main_question}\nCandidate's response: {candidate_response}"}
    ]

    messages.append({"role": "system", "content": system_prompt})

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"}
        )
        turn_action = json.loads(response.choices[0].message.content)
        return turn_action
    except openai.APIError as e:
        st.error(f"OpenAI API error during interview turn: {e}")
        return None
    except json.JSONDecodeError:
        st.error(
            "Failed to parse JSON response for interview turn. Please check the API response for errors or try again.")
        st.write(
            f"Raw response (if available): {response.choices[0].message.content if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during interview turn: {e}")
        return None


def analyze_interview_transcript(full_transcript, initial_plan):
    st.info("Analyzing your interview performance and generating a detailed report...")
    try:
        messages = [
            {"role": "system", "content": f"""You are an AI Interview Analyst, providing feedback to a candidate who just completed a rigorous practice interview.
             You have the full transcript of the interview and the initial interview plan (which includes JD/Resume analysis and identified "relevant_expert_terms").
             Your task is to:
             1. Evaluate the candidate's overall performance against the job description requirements and considering the rigor of the interview questions asked.
             2. Identify the candidate's strong suits based on their answers and how well they handled challenging questions. **Specifically, note instances where the candidate effectively used the identified "relevant_expert_terms" (from the initial plan) to demonstrate expertise.**
             3. Identify the candidate's weak suits or areas for improvement, specifically noting if they struggled with probing questions, provided vague answers, or showed inconsistencies. **Also, highlight opportunities where the candidate *could have used* relevant "expert terms" but did not, or used them incorrectly, impacting their perceived expertise.**
             4. Provide a clear recommendation on whether the candidate is well-prepared for the position (e.g., "Highly Prepared", "Well Prepared", "Needs More Practice", "Significant Improvement Needed").
             5. Give a concise, professional summary of the interview and key takeaways for the candidate's preparation.
             Initial interview plan details for context: {json.dumps(initial_plan, indent=2)}
             Provide the output in a structured JSON format with keys: "overall_qualification",
             "strong_suits", "weak_suits", "recommendation", "interview_summary"."""},
        ]

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"}
        )
        analysis_results = json.loads(response.choices[0].message.content)
        return analysis_results
    except openai.APIError as e:
        st.error(f"OpenAI API error during final analysis: {e}")
        return None
    except json.JSONDecodeError:
        st.error(
            "Failed to parse JSON response for final analysis. Please check the API response for errors or try again.")
        st.write(
            f"Raw response (if available): {response.choices[0].message.content if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during final analysis: {e}")
        return None


# --- Streamlit UI Configuration ---
st.set_page_config(page_title="AI Interview Prep Tool", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F0F2F6; /* Light gray background */
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.1rem;
        font-weight: bold;
    }
    .css-1d391kg { /* Target for header text */
        font-family: 'Segoe UI', sans-serif;
        color: #2F4F4F; /* Dark Slate Gray */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Session State Initialization ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "user_credits" not in st.session_state:  # New: User credits
    st.session_state.user_credits = 0  # Default to 0 until logged in
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "interview_plan" not in st.session_state:
    st.session_state.interview_plan = None
if "current_question_idx" not in st.session_state:
    st.session_state.current_question_idx = 0
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "full_transcript" not in st.session_state:
    st.session_state.full_transcript = ""
if "interview_completed" not in st.session_state:
    st.session_state.interview_completed = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "current_ai_question" not in st.session_state:  # To persist AI's question across reruns
    st.session_state.current_ai_question = ""
if "main_questions_asked_count" not in st.session_state:  # To track how many main questions have been asked
    st.session_state.main_questions_asked_count = 0
if "show_plan_ready_message" not in st.session_state:
    st.session_state.show_plan_ready_message = False


# --- Login Page Function ---
def show_login_page():
    st.title("🔐 Login to AI Interview Prep Tool")
    # st.markdown("""
    # <p style='font-size: 1.1em; color: #555;'>
    # Please log in to access the interview preparation features.
    # <br>
    # <br>
    # For this demo, you can use these credentials:
    # <ul>
    #     <li><b>Username:</b> <code>Admin</code> | <b>Password:</b> <code>Admin123</code> (Unlimited Credits)</li>
    #     <li><b>Username:</b> <code>candidate1</code> | <b>Password:</b> <code>pass123</code> (50 Credits)</li>
    #     <li><b>Username:</b> <code>user_test</code> | <b>Password:</b> <code>testpass</code> (20 Credits)</li>
    # </ul>
    # </p>
    # """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if username in PREDEFINED_USERS and PREDEFINED_USERS[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_credits = PREDEFINED_USERS[username]["initial_credits"]  # Set initial credits
                st.success(
                    f"Welcome, {username}! You have {st.session_state.user_credits if st.session_state.user_credits != -1 else 'unlimited'} credits.")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.markdown("---")


# --- Main Application Logic ---
if not st.session_state.logged_in:
    show_login_page()
else:
    st.title("📚 AI Interview Prep Tool")

    # Display current user and credits
    col_user, col_credits, col_logout = st.columns([0.4, 0.4, 0.2])
    with col_user:
        st.markdown(f"**Logged in as:** `{st.session_state.username}`")
    with col_credits:
        if st.session_state.user_credits == -1:
            st.markdown("**Credits:** Unlimited")
        else:
            st.markdown(f"**Credits:** `{st.session_state.user_credits}`")
    with col_logout:
        if st.button("Logout", key="logout_btn"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_credits = 0  # Reset credits on logout
            # Clear other session state variables related to interview if user logs out
            st.session_state.interview_started = False
            st.session_state.interview_plan = None
            st.session_state.current_question_idx = 0
            st.session_state.conversation_history = []
            st.session_state.full_transcript = ""
            st.session_state.interview_completed = False
            st.session_state.analysis_results = None
            st.session_state.current_ai_question = ""
            st.session_state.main_questions_asked_count = 0
            st.rerun()

    # Simple Credit Top-up for demo
    if st.session_state.user_credits != -1:  # Only show top-up for users with limited credits
        if st.button(f"✨ Top Up {INTERVIEW_COST * 2} Credits",
                     help=f"Add {INTERVIEW_COST * 2} credits to your account for this session."):
            st.session_state.user_credits += (INTERVIEW_COST * 2)
            st.success(f"Credits topped up! You now have {st.session_state.user_credits} credits.")
            st.rerun()

    st.markdown("---")

    # --- Tabbed Interface ---
    tabs = st.tabs(["🚀 Prepare for Interview", "🗣️ Practice Interview", "📊 Get Analysis"])

    with tabs[0]:  # Setup Interview Tab
        st.header("1. Upload Documents & Generate Plan")
        col1, col2 = st.columns(2)
        with col1:
            jd_file = st.file_uploader("Upload Job Description (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"],
                                       key="jd_upload",
                                       help="Upload the job description for the role you are interviewing for.")
        with col2:
            resume_file = st.file_uploader("Upload Your Resume (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"],
                                           key="resume_upload", help="Upload your resume for personalized practice.")

        if st.session_state.interview_started:  # If an interview is already ongoing
            st.info("An interview is already in progress. Please complete it or restart the app to begin a new one.")
            if st.button("Reset Interview Session (Will not refund credits for ongoing session)"):
                # Reset only interview-related session state vars, not login/credits
                st.session_state.interview_started = False
                st.session_state.interview_plan = None
                st.session_state.current_question_idx = 0
                st.session_state.conversation_history = []
                st.session_state.full_transcript = ""
                st.session_state.interview_completed = False
                st.session_state.analysis_results = None
                st.session_state.current_ai_question = ""
                st.session_state.main_questions_asked_count = 0
                st.session_state.show_plan_ready_message = False
                st.rerun()
        if st.session_state.show_plan_ready_message and st.session_state.interview_started and not st.session_state.interview_completed:
            st.success(
                "✨ Your practice interview plan is ready! Please click on the **'🗣️ Practice Interview'** tab above to begin the simulation.")
            st.write("---")  # Add a separator for better UI if needed

        if jd_file and resume_file and not st.session_state.interview_started:
            jd_text = process_uploaded_file(jd_file)
            resume_text = process_uploaded_file(resume_file)

            if jd_text and resume_text:
                st.success("✅ Documents uploaded and processed successfully!")
                if st.button(f"Generate Interview Plan (Costs {INTERVIEW_COST} Credits)"):
                    # --- Credit Check Before Generating Plan ---
                    if st.session_state.user_credits == -1 or st.session_state.user_credits >= INTERVIEW_COST:
                        with st.spinner(
                                "⏳ Analyzing documents and preparing your practice interview plan... This may take a moment."):
                            interview_plan = analyze_documents_and_prepare_for_interview(jd_text, resume_text)
                            if interview_plan:
                                # Deduct credits if not Admin
                                if st.session_state.user_credits != -1:
                                    st.session_state.user_credits -= INTERVIEW_COST
                                    st.success(
                                        f"Credits deducted. You now have {st.session_state.user_credits} credits.")

                                st.session_state.interview_plan = interview_plan
                                st.session_state.interview_started = True
                                st.session_state.conversation_history = [  # Reset history for new interview
                                    {"role": "system",
                                     "content": "Interview started based on JD and Resume. Focus on rigorous testing."},
                                    {"role": "system", "content": json.dumps(interview_plan)}
                                    # Include plan in system history for AI context
                                ]
                                st.session_state.current_question_idx = 0
                                st.session_state.main_questions_asked_count = 0
                                st.session_state.interview_completed = False
                                st.session_state.analysis_results = None

                                # Start with the first main question
                                if interview_plan["interview_questions"]:
                                    first_question = interview_plan["interview_questions"][0]["question"]
                                    st.session_state.current_ai_question = first_question
                                    st.session_state.conversation_history.append(
                                        {"role": "assistant", "content": first_question})
                                    st.session_state.main_questions_asked_count += 1
                                else:
                                    st.session_state.current_ai_question = "No questions generated. Please check plan."
                                    st.error(
                                        "No practice interview questions were generated. Please refine input or model response.")

                                st.session_state.show_plan_ready_message = True
                                st.write("---")
                                st.subheader("📝 Your Initial Practice Questions:")
                                for i, q_data in enumerate(interview_plan["interview_questions"]):
                                    st.markdown(f"**Question {i + 1}:** {q_data['question']}")
                                    with st.expander("Expected Answer Insight"):
                                        st.write(q_data['expected_answer_insight'])
                                st.write("---")

                                st.rerun()
                    else:
                        st.error(
                            f"You need at least {INTERVIEW_COST} credits to generate an interview plan. Your current credits: {st.session_state.user_credits}. Please top up your credits.")

    with tabs[1]:  # Conduct Interview Tab
        st.header("2. Practice Your Interview")
        if st.session_state.interview_started and st.session_state.interview_plan and st.session_state.show_plan_ready_message:
            st.session_state.show_plan_ready_message = False

        if not st.session_state.interview_started:
            st.warning(
                "Please upload documents and generate the interview plan in the '🚀 Prepare for Interview' tab first.")
        elif st.session_state.interview_completed:
            st.info("Your practice interview has concluded. Please view your analysis in the '📊 Get Analysis' tab.")
        elif st.session_state.interview_plan:
            questions_from_plan = st.session_state.interview_plan["interview_questions"]

            # Display AI's current question
            if st.session_state.current_ai_question:
                # In a full system, you'd use OpenAI TTS here to speak the question.
                # Example: audio_output = openai.audio.speech.create(model="tts-1", voice="nova", input=st.session_state.current_ai_question)
                # st.audio(audio_output.read(), format="audio/mp3", autoplay=True)
                st.markdown(f"### 🗣️ AI Interviewer: {st.session_state.current_ai_question}")
            else:
                st.warning("AI is ready to ask a question, but no question is set. This might be an internal error.")

            st.markdown("---")
            st.subheader("Your Response:")

            # --- Voice Recording Integration ---
            # Audio recorder component
            # The default audio format for st_audio_recorder is WAV. Whisper supports WAV.
            audio_bytes = st_audiorecorder(key="audio_recorder")

            candidate_response_input = ""
            if audio_bytes:
                # FIX 3: Added logic to handle AudioSegment if returned by st_audiorecorder
                # This fixes "RuntimeError: Invalid binary data format: <class 'pydub.audio_segment.AudioSegment'>"
                if isinstance(audio_bytes, AudioSegment):
                    buffer = io.BytesIO()
                    # Export to WAV format (ensure you have FFmpeg installed for this)
                    audio_bytes.export(buffer, format="wav")
                    display_audio_bytes = buffer.getvalue()
                else:
                    # If it's already bytes (as expected from st_audiorecorder), use it directly
                    display_audio_bytes = audio_bytes

                st.audio(display_audio_bytes, format="audio/wav")  # Play back recorded audio for user confirmation

                with st.spinner("Transcribing your voice using OpenAI Whisper..."):
                    try:
                        buffer = io.BytesIO()
                        audio_bytes.export(buffer, format="wav")
                        buffer.seek(0)
                        buffer.name = "candidate_response.wav"
                        transcription = openai.audio.transcriptions.create(model="whisper-1", file=buffer)
                        candidate_response_input = transcription.text
                        st.success("Transcription Complete!")
                        st.write(f"**Your Transcribed Response:** {candidate_response_input}")
                    except openai.APIError as e:
                        st.error(f"Failed to transcribe audio with Whisper: {e}")
                    except Exception as e:
                        st.error(f"An error occurred during audio transcription: {e}")

            # Fallback text area (or primary if audio_bytes is empty)
            if not candidate_response_input:  # If no audio was recorded/transcribed, allow text input
                candidate_response_input = st.text_area(
                    "Or type your answer here (if not using voice or for fallback):",
                    key="candidate_response_text_area", height=100)

            # --- End Voice Recording Integration ---

            if st.button("Submit My Response and Get AI's Next Question"):
                if candidate_response_input:  # Check if either text or transcribed audio response is available
                    # Add current interaction to full transcript
                    st.session_state.full_transcript += f"AI: {st.session_state.current_ai_question}\nCandidate: {candidate_response_input}\n"

                    # Update conversation history for AI's context
                    st.session_state.conversation_history.append(
                        {"role": "assistant", "content": st.session_state.current_ai_question})
                    st.session_state.conversation_history.append({"role": "user", "content": candidate_response_input})

                    with st.spinner("🧠 AI thinking, analyzing your response, and formulating the next challenge..."):
                        turn_action = conduct_interview_turn(
                            st.session_state.conversation_history,
                            st.session_state.current_ai_question,
                            candidate_response_input,
                            st.session_state.interview_plan  # Pass full plan for richer context in prompting
                        )

                        if turn_action:
                            # Log internal assessment in history for final analysis context
                            st.session_state.conversation_history.append({"role": "assistant",
                                                                          "content": f"Internal Assessment: {turn_action.get('internal_assessment_of_last_response', 'N/A')}"})

                            st.success("Response processed!")
                            st.markdown(
                                f"**AI's Internal Assessment of Your Last Response:** {turn_action.get('internal_assessment_of_last_response', 'N/A')}")
                            st.markdown(
                                f"**AI's Reason for Next Action:** {turn_action.get('reason_for_action', 'N/A')}")
                            st.write("---")

                            if turn_action["action"] == "ask_follow_up":
                                st.session_state.current_ai_question = turn_action["next_question"]
                                st.rerun()
                            elif turn_action["action"] == "ask_next_question":
                                st.session_state.current_question_idx += 1  # Move to next main question in the plan
                                st.session_state.main_questions_asked_count += 1
                                if st.session_state.current_question_idx < len(questions_from_plan):
                                    st.session_state.current_ai_question = \
                                    questions_from_plan[st.session_state.current_question_idx]["question"]
                                    st.rerun()
                                else:
                                    st.session_state.interview_completed = True
                                    st.info(
                                        "🎉 All main questions from the plan have been asked and thoroughly explored. Your practice interview is complete. Please view your analysis in the '📊 Get Analysis' tab.")
                                    st.rerun()
                            elif turn_action["action"] == "conclude_interview":
                                st.session_state.interview_completed = True
                                st.info(
                                    "🎉 Your practice interview has concluded by AI. Please navigate to the '📊 Get Analysis' tab for your results!")
                                st.rerun()
                        else:
                            st.error("Failed to get a valid action from AI. Please try again.")
                else:
                    st.warning("Please record your response or type your answer to continue the interview.")
        else:
            st.info("AI Interview Agent is ready to proceed to the next question or conclude. Click 'Submit Response'.")
            # This branch ensures the button is available even if AI's last action wasn't a question (e.g., if it decided to end)

    with tabs[2]:  # View Analysis Tab
        st.header("3. Get Your Interview Analysis & Report")

        if not st.session_state.interview_completed:
            st.info(
                "Please complete your practice interview in the '🗣️ Practice Interview' tab to view your analysis.")
        elif not st.session_state.analysis_results:
            st.warning(
                "Your interview is complete! Click the button below to generate your personalized analysis report.")
            if st.button("Generate Interview Analysis Report"):
                with st.spinner("📊 Analyzing your full interview transcript and generating report..."):
                    analysis_results = analyze_interview_transcript(st.session_state.full_transcript,
                                                                    st.session_state.interview_plan)
                    if analysis_results:
                        st.session_state.analysis_results = analysis_results
                        st.success("✅ Analysis complete!")
                        st.rerun()
        else:
            results = st.session_state.analysis_results

            st.subheader("🌟 Overall Preparation Level:")
            st.success(results.get("overall_qualification", "N/A"))

            st.subheader("💪 Your Strong Suits (Based on this practice):")
            strong_suits = results.get("strong_suits", [])
            if strong_suits:
                for suit in strong_suits:
                    if isinstance(suit, dict):  # Handle cases where it might be a dict e.g. {"suit": "detail"}
                        st.markdown(f"- {list(suit.values())[0]}")
                    else:
                        st.markdown(f"- {suit}")
            else:
                st.write("N/A")

            st.subheader("🚧 Areas for Improvement (Practice More Here):")
            # FIX 4: Removed the stray "" that was causing a Pylance error
            weak_suits = results.get("weak_suits", [])
            if weak_suits:
                # Assuming the structure shown in image_78cc88.png
                for i, suit in enumerate(weak_suits):
                    if isinstance(suit, dict):
                        st.write(f"* {suit.get(str(i), 'N/A')}")
                    else:
                        st.write(f"* {suit}")
            else:
                st.write("N/A")

            st.subheader("💡 Recommendation:")
            st.info(results.get("recommendation", "N/A"))

            st.subheader("Summary & Key Takeaways:")
            st.markdown(results.get("interview_summary", "N/A"))

            st.subheader("Full Transcript of Your Interview:")
            st.text_area("Read through your full conversation with the AI:",
                         value=st.session_state.full_transcript, height=300)
