"""Analysis Results page with score cards, charts, skill gap tables, and interview prep."""

from __future__ import annotations

import json
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from config import can_access_admin_settings, get_config
from database.db import get_user_analysis, list_user_analyses
from modules.resume_analyzer import ATS_DISCLAIMER


def _build_markdown_report(job_title: str, resume_filename: str, report: dict[str, Any]) -> str:
    """Format an analysis report as a clean downloadable Markdown document."""
    ats = report.get("ats_evaluation", {})
    comp = report.get("compatibility", {})
    gap = report.get("skill_gap", {})
    iv = report.get("interview_questions", {})

    lines = [
        f"# VYRAQ AI Career Compatibility Report — {job_title}",
        f"**Resume:** {resume_filename}",
        f"**Executive Summary:** {report.get('executive_summary', '')}",
        "",
        f"## A. ATS Resume Score: {ats.get('ats_score', 0)} / 100",
        f"> {ats.get('ats_disclaimer', ATS_DISCLAIMER)}",
        "",
        f"**Quality Assessment:** {ats.get('resume_quality_assessment', '')}",
        f"- **Detected Keywords:** {', '.join(ats.get('important_keywords_detected', []))}",
        f"- **Missing Keywords:** {', '.join(ats.get('missing_keywords', []))}",
        "",
        "### ATS Improvement Suggestions",
    ]
    for sug in ats.get("ats_suggestions", []):
        lines.append(f"- {sug}")

    lines.extend(
        [
            "",
            f"## B. Job Compatibility Score: {comp.get('compatibility_score', 0)}%",
            f"**Explanation:** {comp.get('match_explanation', '')}",
            f"**Experience Comparison:** {comp.get('experience_comparison', '')}",
            f"- **Matching Skills:** {', '.join(comp.get('matching_skills', []))}",
            f"- **Partially Matching Skills:** {', '.join(comp.get('partially_matching_skills', []))}",
            f"- **Missing Skills:** {', '.join(comp.get('missing_skills', []))}",
            "",
            "## C. Skill Gap Analysis",
            f"- **Required Skills:** {', '.join(gap.get('required_skills', []))}",
            f"- **Available Skills:** {', '.join(gap.get('available_skills', []))}",
            f"- **Missing Skills:** {', '.join(gap.get('missing_skills', []))}",
            "",
            "## D. Tailored Interview Questions",
        ]
    )
    for cat_key, cat_label in (
        ("technical_questions", "Technical"),
        ("role_specific_questions", "Role-Specific"),
        ("resume_based_questions", "Resume-Based"),
        ("hr_questions", "HR"),
    ):
        lines.append(f"### {cat_label} Questions")
        for idx, item in enumerate(iv.get(cat_key, []), start=1):
            lines.append(f"{idx}. **{item.get('question', '')}**")
            lines.append(f"   - *Sample Answer:* {item.get('sample_answer', '')}")
    return "\n".join(lines)


def render_results_page() -> None:
    """Render detailed interactive analysis results for the selected report."""
    cfg = get_config()
    user_id = int(st.session_state["user_id"])
    selected_id = st.session_state.get("selected_analysis_id")

    record = get_user_analysis(cfg.db_path, user_id, int(selected_id)) if selected_id else None
    if record is None:
        recent = list_user_analyses(cfg.db_path, user_id, limit=1)
        if recent:
            record = recent[0]
            st.session_state["selected_analysis_id"] = record.id

    if record is None:
        st.title("Analysis Results")
        st.info("No analysis reports found yet. Run your first analysis on the **Upload & Analyze** page.")
        if st.button("Go to Upload & Analyze", type="primary"):
            st.session_state["current_page"] = "Upload & Analyze"
            st.rerun()
        return

    report = record.report_data
    ats = report.get("ats_evaluation", {})
    ats_break = ats.get("ats_breakdown", {})
    comp = report.get("compatibility", {})
    comp_break = comp.get("breakdown", {})
    gap = report.get("skill_gap", {})
    iv = report.get("interview_questions", {})
    rag_meta = report.get("rag_metadata", {})

    is_admin = can_access_admin_settings(st.session_state.get("user_email", ""), cfg)
    st.title(f"Analysis Report: {record.job_title}")
    if is_admin:
        st.caption(
            f"Resume: `{record.resume_filename}` • Provider: `{record.provider}/{record.model_name}` • Created: {record.created_at}"
        )
    else:
        st.caption(f"Resume: `{record.resume_filename}` • Created: {record.created_at}")

    if report.get("executive_summary"):
        st.info(report["executive_summary"])

    # Top Score Cards
    k1, k2, k3, k4 = st.columns(4)
    matched_cnt = len(comp.get("matching_skills", []))
    partial_cnt = len(comp.get("partially_matching_skills", []))
    missing_cnt = len(gap.get("missing_skills", comp.get("missing_skills", [])))

    k1.metric("ATS Resume Score", f"{record.ats_score:.1f} / 100")
    k2.metric("Job Compatibility", f"{record.compatibility_score:.1f}%")
    k3.metric("Matched Skills", str(matched_cnt), delta=f"+{partial_cnt} partial" if partial_cnt else None)
    k4.metric("Missing Skills", str(missing_cnt))

    # Download buttons
    dl1, dl2, _ = st.columns([1, 1, 2])
    with dl1:
        st.download_button(
            "Download JSON Report",
            data=json.dumps(report, indent=2),
            file_name=f"vyraq_report_{record.id}.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl2:
        md_text = _build_markdown_report(record.job_title, record.resume_filename, report)
        st.download_button(
            "Download Markdown Summary",
            data=md_text,
            file_name=f"vyraq_report_{record.id}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    tab_ats, tab_comp, tab_gap, tab_iv = st.tabs(
        [
            "ATS Score & Keywords",
            "Job Compatibility",
            "Skill Gap Analysis",
            "Interview Questions",
        ]
    )

    # TAB A: ATS Resume Score
    with tab_ats:
        st.subheader("ATS Resume Score Breakdown")
        st.caption(ats.get("ats_disclaimer", ATS_DISCLAIMER))

        b1, b2 = st.columns(2)
        with b1:
            kw_score = float(ats_break.get("keyword_relevance", 0.0))
            imp_score = float(ats_break.get("quantified_impact", 0.0))
            st.markdown(f"**Keyword Relevance:** {kw_score:.1f} / 35")
            st.progress(min(1.0, max(0.0, kw_score / 35.0)))
            st.markdown(f"**Quantified Impact & Action Verbs:** {imp_score:.1f} / 25")
            st.progress(min(1.0, max(0.0, imp_score / 25.0)))
        with b2:
            sec_score = float(ats_break.get("section_structure", 0.0))
            clr_score = float(ats_break.get("clarity_formatting", 0.0))
            st.markdown(f"**Section Completeness:** {sec_score:.1f} / 20")
            st.progress(min(1.0, max(0.0, sec_score / 20.0)))
            st.markdown(f"**Clarity & Formatting:** {clr_score:.1f} / 20")
            st.progress(min(1.0, max(0.0, clr_score / 20.0)))

        st.markdown("#### Resume Quality Assessment")
        st.write(ats.get("resume_quality_assessment", "No assessment available."))

        kw_col1, kw_col2 = st.columns(2)
        with kw_col1:
            st.markdown("#### Important Keywords Detected")
            detected_kw = ats.get("important_keywords_detected", [])
            if detected_kw:
                st.dataframe(pd.DataFrame({"Detected Keyword": detected_kw}), use_container_width=True, hide_index=True)
            else:
                st.write("None listed.")
        with kw_col2:
            st.markdown("#### Missing Keywords")
            missing_kw = ats.get("missing_keywords", [])
            if missing_kw:
                st.dataframe(pd.DataFrame({"Missing Keyword": missing_kw}), use_container_width=True, hide_index=True)
            else:
                st.success("No major missing keywords detected.")

        st.markdown("#### Suggestions to Improve ATS Compatibility")
        for sug in ats.get("ats_suggestions", []):
            st.markdown(f"- {sug}")

    # TAB B: Job Compatibility
    with tab_comp:
        st.subheader("Job Compatibility Breakdown")
        c_left, c_right = st.columns([1.1, 1.2])
        with c_left:
            req_s = float(comp_break.get("required_skills_score", 0.0))
            pref_s = float(comp_break.get("preferred_skills_score", 0.0))
            exp_s = float(comp_break.get("experience_alignment_score", 0.0))
            dom_s = float(comp_break.get("domain_keyword_score", 0.0))

            st.markdown(f"**Required Skills Match:** {req_s:.1f} / 50")
            st.progress(min(1.0, max(0.0, req_s / 50.0)))
            st.markdown(f"**Preferred Skills Match:** {pref_s:.1f} / 15")
            st.progress(min(1.0, max(0.0, pref_s / 15.0)))
            st.markdown(f"**Experience Alignment:** {exp_s:.1f} / 25")
            st.progress(min(1.0, max(0.0, exp_s / 25.0)))
            st.markdown(f"**Domain Terminology Alignment:** {dom_s:.1f} / 10")
            st.progress(min(1.0, max(0.0, dom_s / 10.0)))

        with c_right:
            chart_df = pd.DataFrame(
                {
                    "Category": ["Matching Skills", "Partial Skills", "Missing Skills"],
                    "Count": [
                        len(comp.get("matching_skills", [])),
                        len(comp.get("partially_matching_skills", [])),
                        len(comp.get("missing_skills", [])),
                    ],
                }
            )
            chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Category:N", sort=None, title=""),
                    y=alt.Y("Count:Q", title="Skill Count"),
                    color=alt.Color(
                        "Category:N",
                        scale=alt.Scale(
                            domain=["Matching Skills", "Partial Skills", "Missing Skills"],
                            range=["#10B981", "#F59E0B", "#EF4444"],
                        ),
                        legend=None,
                    ),
                    tooltip=["Category", "Count"],
                )
                .properties(height=240, title="Skill Alignment Overview")
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown("#### Why You Match / Gaps to Address")
        st.write(comp.get("match_explanation", ""))
        st.markdown("#### Relevant Experience Comparison")
        st.write(comp.get("experience_comparison", ""))

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**Matching Skills**")
            for s in comp.get("matching_skills", []):
                st.markdown(f"- :green[{s}]")
        with s2:
            st.markdown("**Partially Matching Skills**")
            for s in comp.get("partially_matching_skills", []):
                st.markdown(f"- :orange[{s}]")
        with s3:
            st.markdown("**Missing Skills**")
            for s in comp.get("missing_skills", []):
                st.markdown(f"- :red[{s}]")

    # TAB C: Skill Gap Analysis
    with tab_gap:
        st.subheader("Skill Gap Matrix & Learning Roadmap")
        if gap.get("gap_summary"):
            st.write(gap["gap_summary"])

        req_set = gap.get("required_skills", [])
        avail_set = {s.lower() for s in gap.get("available_skills", [])}
        partial_set = {s.lower() for s in comp.get("partially_matching_skills", [])}
        matrix_rows = []
        for skill in req_set:
            low = skill.lower()
            status = "Available" if low in avail_set else ("Partial" if low in partial_set else "Missing")
            matrix_rows.append({"Required Skill": skill, "Resume Status": status})
        if matrix_rows:
            st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

        recs = gap.get("recommended_skills_to_learn", [])
        if recs:
            st.markdown("#### Recommended Skills to Learn")
            rec_df = pd.DataFrame(
                [
                    {
                        "Skill": r.get("skill", ""),
                        "Priority": r.get("priority", "Medium"),
                        "Reason": r.get("reason", ""),
                        "Recommended Focus": r.get("learning_focus", ""),
                    }
                    for r in recs
                ]
            )
            st.dataframe(rec_df, use_container_width=True, hide_index=True)

    # TAB D: Interview Questions
    with tab_iv:
        st.subheader("Tailored Interview Questions & Sample Answers")
        categories = [
            ("technical_questions", "Technical Questions"),
            ("role_specific_questions", "Role-Specific Questions"),
            ("resume_based_questions", "Resume-Based Questions"),
            ("hr_questions", "HR & Behavioral Questions"),
        ]
        for key, heading in categories:
            items = iv.get(key, [])
            st.markdown(f"### {heading}")
            if not items:
                st.caption("No questions generated in this category.")
                continue
            for idx, q in enumerate(items, start=1):
                title = f"{idx}. [{q.get('difficulty', 'Medium')}] {q.get('question', '')}"
                with st.expander(title, expanded=(idx == 1 and key == "technical_questions")):
                    if q.get("target_skill_or_topic"):
                        st.caption(f"Target Topic: **{q['target_skill_or_topic']}**")
                    if q.get("rationale"):
                        st.markdown(f"**Why Asked:** {q['rationale']}")
                    if q.get("key_answer_points"):
                        st.markdown("**Key Answer Points:**")
                        for pt in q["key_answer_points"]:
                            st.markdown(f"- {pt}")
                    if q.get("sample_answer"):
                        st.markdown("**Sample Answer:**")
                        st.write(q["sample_answer"])

    if rag_meta and is_admin:
        with st.expander("Pipeline & RAG Grounding Telemetry"):
            st.json(
                {
                    "rag_metadata": rag_meta,
                    "workflow_steps_executed": report.get("workflow_steps", []),
                }
            )
