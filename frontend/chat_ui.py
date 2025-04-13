import streamlit as st
import requests

API_ENDPOINT = "http://localhost:8000/query"  # Update if hosted elsewhere

st.set_page_config(page_title="Resume Chatbot", layout="wide")
st.title("💬 Resume Chatbot Interface")

# --- Input section ---
query = st.text_input("Enter your query", placeholder="e.g., Who knows Python and PHP both?")
use_gemini = st.checkbox("Use Gemini LLM", value=False)

def render_resume_markdown(resume: dict) -> str:
    """Convert resume JSON into markdown string"""
    parts = []

    parts.append(f"### 👤 {resume.get('name', 'Unnamed')}")
    if resume.get("email"):
        parts.append(f"- 📧 Email: {resume['email']}")
    if resume.get("phone"):
        parts.append(f"- 📞 Phone: {resume['phone']}")
    if resume.get("location"):
        parts.append(f"- 📍 Location: {resume['location']}")

    if summary := resume.get("summary"):
        parts.append(f"\n**Summary:**\n{summary}")

    if education := resume.get("education"):
        parts.append("\n**🎓 Education:**")
        for edu in education:
            parts.append(f"- {edu.get('degree', '')} | {edu.get('institution', '')} ({edu.get('year', '')})")

    if experience := resume.get("experience"):
        parts.append("\n**💼 Experience:**")
        for exp in experience:
            parts.append(f"- **{exp.get('title', '')}**, {exp.get('company', '')} ({exp.get('duration', '')})")
            if desc := exp.get("description"):
                parts.append(f"  \n  {desc}")

    if skills := resume.get("skills"):
        parts.append("\n**🛠️ Skills:**")
        parts.append(", ".join(skills))

    if projects := resume.get("projects"):
        parts.append("\n**📁 Projects:**")
        for proj in projects:
            parts.append(f"- **{proj.get('title')}**: {proj.get('description')}")

    if certs := resume.get("certifications"):
        parts.append("\n**📜 Certifications:**")
        for cert in certs:
            parts.append(f"- {cert.get('title')} ({cert.get('issuer', '')})")

    return "\n".join(parts)

# --- Submit & Display ---
if st.button("Submit Query"):
    with st.spinner("Processing..."):
        response = requests.post(API_ENDPOINT, json={"query": query, "use_gemini": use_gemini})
        data = response.json()

        if "error" in data:
            st.error(f"Error: {data['error']}")
            if data.get("keywords"):
                st.info(f"Extracted Keywords: {', '.join(data['keywords'])}")
        else:
            st.subheader("🧠 Bot Answer")
            st.markdown(data["answer"])  # Now renders as markdown

            st.subheader("📄 Relevant Resumes")
            st.markdown(f"**Total Relevant Resumes:** {len(data['relevant_resumes'])}")
            for resume in data["relevant_resumes"]:
                with st.expander(resume.get("name", "Unnamed")):
                    st.markdown(render_resume_markdown(resume), unsafe_allow_html=True)
