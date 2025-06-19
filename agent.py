import streamlit as st
import openai
import json
import os
import io
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in your .env file or system environment.")
    st.stop()

openai.api_key = openai_api_key

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

def analyze_documents_and_prepare_for_interview(jd_text, resume_text):
    st.info("Analyzing Job Description and Resume...")
    try:
        messages = [
            {"role": "system", "content": """You are an expert HR Interview AI.
             Your task is to analyze a Job Description and a Candidate's Resume.
             Based on this analysis, you will:
             1. Identify key skills, experiences, and qualifications required by the JD.
             2. Identify the candidate's relevant skills, experiences, and qualifications from their resume.
             3. Pinpoint any gaps or areas where the candidate's resume seems strong or weak relative to the JD.
             4. Generate an initial set of 5-7 interview questions designed to assess the candidate's fit,
                probe into their experiences, and clarify any ambiguities.
             5. For each question, suggest what a good answer might entail or what you're looking for.
             Provide the output in a structured JSON format with keys: "jd_analysis", "resume_analysis",
             "gaps_and_strengths", "interview_questions". The "interview_questions" should be a list of dictionaries,
             each with "question" and "expected_answer_insight" keys."""},
            {"role": "user", "content": f"Job Description:\n{jd_text}\n\nCandidate Resume:\n{resume_text}"}
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
        st.error("Failed to parse JSON response from OpenAI. Retrying or adjusting prompt might be needed.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during planning: {e}")
        return None


def conduct_interview_turn(conversation_history, current_question, candidate_response):
    st.info("Processing interview turn...")
    messages = conversation_history + [
        {"role": "user", "content": f"AI's last question: {current_question}\nCandidate's response: {candidate_response}"}
    ]

    messages.append({
        "role": "system",
        "content": """Based on the previous conversation, the candidate's response to the last question,
        and the overall interview plan (which you generated earlier), decide on the next action.
        You can either:
        1. Ask a follow-up question to clarify or probe deeper into the candidate's last answer.
        2. Ask the next main question from the pre-generated interview plan.
        3. Conclude the interview if enough information has been gathered or if the pre-set number of questions is exhausted.
        Also, provide a brief internal assessment of the candidate's response to the *last question*.
        Return your response in JSON format with keys: "action" (e.g., "ask_follow_up", "ask_next_question", "conclude_interview"),
        "next_question" (if applicable), "internal_assessment_of_last_response", "reason_for_action"."""
    })

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
        st.error("Failed to parse JSON response for interview turn. Retrying or adjusting prompt might be needed.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during interview turn: {e}")
        return None


def analyze_interview_transcript(full_transcript, initial_plan):
    st.info("Analyzing interview transcript...")
    try:
        messages = [
            {"role": "system", "content": f"""You are an AI Interview Analyst.
             You have the full transcript of an interview and the initial interview plan.
             Your task is to:
             1. Evaluate the candidate's overall performance against the job description requirements.
             2. Identify the candidate's strong suits based on their answers and the JD.
             3. Identify the candidate's weak suits or areas for improvement.
             4. Provide a recommendation on whether the candidate is qualified for the position (e.g., "Strongly Recommended", "Recommended", "Not Recommended").
             5. Give a concise summary of the interview.
             Initial interview plan details for context: {json.dumps(initial_plan, indent=2)}
             Provide the output in a structured JSON format with keys: "overall_qualification",
             "strong_suits", "weak_suits", "recommendation", "interview_summary"."""},
            {"role": "user", "content": f"Full Interview Transcript:\n{full_transcript}"}
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
        st.error("Failed to parse JSON response for final analysis. Retrying or adjusting prompt might be needed.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during final analysis: {e}")
        return None

st.set_page_config(page_title="AI Interview Agent", layout="wide", initial_sidebar_state="expanded")

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

st.title("🤖 AI Interview Agent")
st.markdown("""
<p style='font-size: 1.1em; color: #555;'>
Welcome to your AI Interview Agent! This tool will help you conduct interviews
autonomously. Upload a Job Description and a Candidate's Resume to get started.
</p>
""", unsafe_allow_html=True)


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

tabs = st.tabs(["🚀 Setup Interview", "🗣️ Conduct Interview", "📊 View Analysis"])

with tabs[0]:
    st.header("1. Upload Documents & Plan Interview")
    col1, col2 = st.columns(2)
    with col1:
        jd_file = st.file_uploader("Upload Job Description (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"], key="jd_upload", help="Upload the job description for the role you are interviewing for.")
    with col2:
        resume_file = st.file_uploader("Upload Candidate Resume (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"], key="resume_upload", help="Upload the candidate's resume.")

    if jd_file and resume_file:
        jd_text = process_uploaded_file(jd_file)
        resume_text = process_uploaded_file(resume_file)

        if jd_text and resume_text:
            st.success("✅ Documents uploaded and processed successfully!")
            if st.button("Prepare Interview Plan"):
                with st.spinner("⏳ Preparing interview questions and plan... This may take a moment."):
                    interview_plan = analyze_documents_and_prepare_for_interview(jd_text, resume_text)
                    if interview_plan:
                        st.session_state.interview_plan = interview_plan
                        st.session_state.interview_started = True
                        st.session_state.conversation_history.append({"role": "system", "content": "Interview started."})
                        st.session_state.conversation_history.append({"role": "system", "content": "Initial interview plan generated based on JD and Resume."})

                        # Emphasized and clearer instruction
                        st.success("✨ Interview plan prepared successfully! Now, please click on the **'🗣️ Conduct Interview'** tab above to begin the simulation.")

                        st.write("---")
                        st.subheader("📝 Initial Interview Questions:")
                        for i, q_data in enumerate(interview_plan["interview_questions"]):
                            st.markdown(f"**Question {i+1}:** {q_data['question']}")
                            with st.expander("Expected Answer Insight"):
                                st.write(q_data['expected_answer_insight'])
                        st.write("---")
                        st.rerun()

with tabs[1]:
    st.header("2. Conduct Interview (Simulated)")

    if not st.session_state.interview_started:
        st.warning("Please upload documents and prepare the interview plan in the '🚀 Setup Interview' tab first.")
    elif st.session_state.interview_completed:
        st.info("The interview has concluded. Please view the analysis in the '📊 View Analysis' tab.")
    elif st.session_state.interview_plan:
        questions = st.session_state.interview_plan["interview_questions"]
        current_idx = st.session_state.current_question_idx

        if current_idx < len(questions):
            current_main_question = questions[current_idx]["question"]
            st.markdown(f"### 🗣️ AI Interviewer: {current_main_question}")

            candidate_response_input = st.text_area("Candidate's Response (Type here):", key=f"response_{current_idx}", height=150)

            if st.button("Submit Response and Get AI's Next Action"):
                if candidate_response_input:
                    st.session_state.full_transcript += f"AI: {current_main_question}\nCandidate: {candidate_response_input}\n"
                    st.session_state.conversation_history.append({"role": "assistant", "content": current_main_question})
                    st.session_state.conversation_history.append({"role": "user", "content": candidate_response_input})

                    with st.spinner("🧠 AI thinking and analyzing response..."):
                        action = conduct_interview_turn(
                            st.session_state.conversation_history,
                            current_main_question,
                            candidate_response_input
                        )

                        if action:
                            st.session_state.conversation_history.append({"role": "assistant", "content": f"Internal Assessment: {action.get('internal_assessment_of_last_response')}"})

                            st.success("Response processed!")
                            st.markdown(f"**AI's Internal Assessment of Last Response:** {action.get('internal_assessment_of_last_response', 'N/A')}")
                            st.markdown(f"**Reason for AI's Action:** {action.get('reason_for_action', 'N/A')}")
                            st.write("---")

                            if action["action"] == "ask_follow_up" or action["action"] == "ask_next_question":
                                st.session_state.conversation_history.append({"role": "assistant", "content": action["next_question"]})
                                st.session_state.current_question_idx += 1
                                st.rerun()
                            elif action["action"] == "conclude_interview":
                                st.session_state.interview_completed = True
                                st.success("🎉 Interview concluded by AI. Please navigate to the '📊 View Analysis' tab for results!")
                                st.rerun()
                else:
                    st.warning("Please enter a candidate response to continue the interview.")
        else:
            st.session_state.interview_completed = True
            st.info("All main questions asked. Interview ending. Please view the analysis in the '📊 View Analysis' tab.")
            st.rerun()

with tabs[2]:
    st.header("3. Interview Analysis & Report")

    if not st.session_state.interview_completed:
        st.info("Please complete the interview in the '🗣️ Conduct Interview' tab to view the analysis.")
    elif not st.session_state.analysis_results:
        st.warning("Analysis not yet generated. Click the button below to start.")
        if st.button("Generate Interview Analysis Report"):
            with st.spinner("📊 Analyzing the full interview transcript and generating report..."):
                analysis_results = analyze_interview_transcript(st.session_state.full_transcript, st.session_state.interview_plan)
                if analysis_results:
                    st.session_state.analysis_results = analysis_results
                    st.success("✅ Analysis complete!")
                    st.rerun()
    else:
        results = st.session_state.analysis_results

        st.subheader("🌟 Overall Qualification:")
        st.success(results.get("overall_qualification", "N/A"))

        st.subheader("💪 Candidate's Strong Suits:")
        strong_suits = results.get("strong_suits", [])
        if strong_suits:
            for suit in strong_suits:
                if isinstance(suit, dict):
                    st.markdown(f"- {list(suit.values())[0]}")
                else:
                    st.markdown(f"- {suit}")
        else:
            st.write("N/A")

        st.subheader("🚧 Candidate's Weak Suits/Areas for Improvement:")
        weak_suits = results.get("weak_suits", [])
        if weak_suits:
            for suit in weak_suits:
                if isinstance(suit, dict):
                    st.markdown(f"- {list(suit.values())[0]}")
                else:
                    st.markdown(f"- {suit}")
        else:
            st.write("N/A")

        st.subheader("✅ Recommendation:")
        st.success(results.get("recommendation", "N/A"))

        st.subheader("📄 Interview Summary:")
        st.info(results.get("interview_summary", "N/A"))

        st.subheader("Full Interview Transcript:")
        with st.expander("Click to view full transcript"):
            st.text_area("Transcript", st.session_state.full_transcript, height=300, disabled=True)
