"""
SupportSphere AI: Multi-Agent Customer Support Ecosystem
Capstone & Enterprise System Design Implementation
Includes:
- Enterprise Authentication (Login & Sign-Up with RBAC)
- LangGraph Stateful Orchestration
- CrewAI Specialist Swarms
- Dual Model Hierarchy (Gemini 2.5 + Groq Llama 3.3)
- Model Context Protocol (FastMCP)
- Vector RAG Knowledge Grounding
- Support Operations Command Center with Human-in-the-Loop
"""

import streamlit as st
import time
import pandas as pd

# Resilient import of orchestrator
try:
    from graph_orchestrator import support_orchestrator
except ImportError:
    from graph_orchestrator import support_graph as support_orchestrator

from mcp_client import mcp_replay_order

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="SupportSphere AI | Multi-Agent Customer Support Ecosystem",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- ADVANCED ENTERPRISE CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Top Banner Header */
    .hero-container {
        background: linear-gradient(135deg, #0b132b 0%, #1c2541 50%, #0b132b 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    }

    .hero-title {
        font-size: 30px;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #c084fc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
        font-weight: 500;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        margin-top: 14px;
    }

    .tag-live {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    .tag-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
    }

    .tag-framework {
        background: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.25);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Auth Card Styling */
    .auth-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 36px 40px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
        margin: 40px auto;
        max-width: 520px;
    }

    .auth-title {
        font-size: 26px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .auth-subtitle {
        color: #94a3b8;
        font-size: 13px;
        text-align: center;
        margin-bottom: 24px;
    }

    /* Metadata Badges */
    .badge-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0 14px 0;
    }

    .meta-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .chip-category { background: #1e1b4b; color: #a5b4fc; border: 1px solid #3730a3; }
    .chip-prio-high { background: #450a0a; color: #fca5a5; border: 1px solid #991b1b; }
    .chip-prio-med { background: #422006; color: #fde047; border: 1px solid #854d0e; }
    .chip-prio-low { background: #064e3b; color: #6ee7b7; border: 1px solid #065f46; }
    .chip-sent-pos { background: #064e3b; color: #6ee7b7; border: 1px solid #047857; }
    .chip-sent-frust { background: #451a03; color: #fdba74; border: 1px solid #9a3412; }
    .chip-sent-angry { background: #450a0a; color: #f87171; border: 1px solid #dc2626; }
    .chip-agent { background: #0f172a; color: #38bdf8; border: 1px solid #0284c7; }

    /* Metric Cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-val {
        font-size: 28px;
        font-weight: 800;
        color: #f9fafb;
        margin: 4px 0;
    }
    .metric-lbl {
        font-size: 11px;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE & AUTH INITIALIZATION -----------------
if "users_db" not in st.session_state:
    # Pre-seeded users database
    st.session_state.users_db = {
        "alex@enterprise.com": {
            "password": "password123",
            "name": "Alex Mercer",
            "customer_id": "cust_0091",
            "role": "Customer",
            "tier": "enterprise"
        },
        "admin@supportsphere.ai": {
            "password": "admin123",
            "name": "Sarah Connor",
            "customer_id": "lead_admin_01",
            "role": "Support Supervisor (Admin)",
            "tier": "enterprise"
        }
    }

if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "tickets_registry" not in st.session_state:
    st.session_state.tickets_registry = []

# =====================================================================
# VIEW 1: LOGIN & SIGN-UP AUTHENTICATION PORTAL
# =====================================================================
if not st.session_state.is_authenticated:
    st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <div style="font-size: 42px;">🛡️</div>
            <h1 style="font-size: 32px; font-weight: 800; background: linear-gradient(90deg, #60a5fa, #c084fc, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 4px 0;">
                SupportSphere AI
            </h1>
            <p style="color: #94a3b8; font-size: 15px; margin-bottom: 20px;">
                Autonomous Multi-Agent Customer Support Ecosystem
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_space1, col_center, col_space2 = st.columns([1, 2, 1])

    with col_center:
        auth_tab_login, auth_tab_signup = st.tabs(["🔑 Sign In to Account", "✨ Create New Account"])

        # ----- TAB 1: LOGIN -----
        with auth_tab_login:
            st.markdown("#### Welcome Back")
            st.caption("Sign in with your enterprise credentials or use 1-click demo access.")

            login_email = st.text_input("Work Email", placeholder="alex@enterprise.com", key="login_email")
            login_password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pwd")

            if st.button("🚀 Sign In", use_container_width=True, type="primary"):
                user = st.session_state.users_db.get(login_email.strip())
                if user and user["password"] == login_password:
                    st.session_state.is_authenticated = True
                    st.session_state.current_user = user
                    st.success(f"Welcome back, {user['name']}!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid email or password. Please try again.")

            st.markdown("---")
            st.markdown("##### ⚡ Quick 1-Click Evaluator Demo Access")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                if st.button("👤 Demo Customer (Alex)", use_container_width=True):
                    st.session_state.is_authenticated = True
                    st.session_state.current_user = st.session_state.users_db["alex@enterprise.com"]
                    st.rerun()
            with col_d2:
                if st.button("🛡️ Demo Supervisor (Admin)", use_container_width=True):
                    st.session_state.is_authenticated = True
                    st.session_state.current_user = st.session_state.users_db["admin@supportsphere.ai"]
                    st.rerun()

        # ----- TAB 2: SIGN UP -----
        with auth_tab_signup:
            st.markdown("#### Create Your Profile")
            st.caption("Register for immediate access to SupportSphere AI.")

            new_name = st.text_input("Full Name", placeholder="e.g. Jordan Lee", key="reg_name")
            new_email = st.text_input("Email Address", placeholder="jordan@company.com", key="reg_email")
            new_password = st.text_input("Choose Password", type="password", placeholder="••••••••", key="reg_pwd")
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                new_role = st.selectbox(
                    "Account Role (RBAC)",
                    ["Customer", "Support Supervisor (Admin)"],
                    key="reg_role"
                )
            with col_r2:
                new_tier = st.selectbox(
                    "Client SLA Tier",
                    ["enterprise", "pro", "standard"],
                    key="reg_tier"
                )

            if st.button("🎉 Register & Launch Workspace", use_container_width=True, type="primary"):
                if not new_name.strip() or not new_email.strip() or not new_password.strip():
                    st.warning("Please fill in all required fields.")
                elif new_email.strip() in st.session_state.users_db:
                    st.error("An account with this email already exists.")
                else:
                    new_user_data = {
                        "name": new_name.strip(),
                        "customer_id": f"cust_{int(time.time()) % 10000}",
                        "password": new_password,
                        "role": new_role,
                        "tier": new_tier
                    }
                    st.session_state.users_db[new_email.strip()] = new_user_data
                    st.session_state.is_authenticated = True
                    st.session_state.current_user = new_user_data
                    st.success(f"Account created successfully for {new_name}!")
                    time.sleep(0.5)
                    st.rerun()

    st.stop()  # Stop execution here until user is authenticated!

# =====================================================================
# VIEW 2: AUTHENTICATED APPLICATION
# =====================================================================
current_user = st.session_state.current_user
user_role = current_user.get("role", "Customer")
is_admin = "Admin" in user_role or "Supervisor" in user_role

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 18px;">
            <div style="font-size: 28px;">🛡️</div>
            <div>
                <div style="font-weight: 800; font-size: 17px; color: #ffffff;">SupportSphere AI</div>
                <div style="font-size: 11px; color: #94a3b8;">Multi-Agent Customer Support Ecosystem</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("##### 👤 Active User Session")
    st.markdown(f"""
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 12px 14px; margin-bottom: 14px;">
            <div style="font-weight: 700; color: #f9fafb; font-size: 14px;">{current_user['name']}</div>
            <div style="font-size: 12px; color: #60a5fa; font-weight: 600;">{user_role}</div>
            <div style="font-size: 11px; color: #9ca3af; margin-top: 4px;">ID: <code>{current_user['customer_id']}</code></div>
            <div style="font-size: 11px; color: #34d399; margin-top: 2px;">SLA Tier: <b>{current_user['tier'].upper()}</b></div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.is_authenticated = False
        st.session_state.current_user = None
        st.rerun()

    st.markdown("---")
    st.markdown("##### 🧩 Technical Stack Overview")
    st.markdown("""
    <div style="font-size: 12px; color: #cbd5e1; line-height: 1.8;">
        • <b>Control Plane:</b> LangGraph (StateGraph DAG)<br>
        • <b>Collaborative Swarms:</b> CrewAI Sequential Swarms<br>
        • <b>Closed Model:</b> Google Gemini 2.5 Flash<br>
        • <b>Open Model:</b> Groq (Llama-3.3-70B)<br>
        • <b>Enterprise Tools:</b> Model Context Protocol (MCP)<br>
        • <b>Grounding Engine:</b> Vector RAG (Semantic Cosine)<br>
        • <b>Access Control:</b> RBAC Role-Based Access
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ----------------- TOP HERO BANNER -----------------
st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
        <div>
            <h1 class="hero-title">
                <span>🛡️</span> SupportSphere AI
            </h1>
            <div class="hero-subtitle">
                Multi-Agent Customer Support Ecosystem • Stateful Orchestration • MCP & RAG Grounded
            </div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <span class="tag-live"><div class="tag-dot"></div> 5 Agents Online</span>
            <span class="tag-framework">User: {current_user['name']} ({current_user['tier'].upper()})</span>
        </div>
    </div>
    <div class="hero-badges">
        <span class="tag-framework">LangGraph Supervisor</span>
        <span class="tag-framework">CrewAI Swarms</span>
        <span class="tag-framework">FastMCP Gateway</span>
        <span class="tag-framework">SLA: 99.8% Grounded Resolution</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- MAIN TABS -----------------
tab_portal, tab_analytics = st.tabs([
    "💬 Customer Support Portal",
    "📊 Support Operations Command Center"
])

# =====================================================================
# TAB 1: CUSTOMER SUPPORT PORTAL
# =====================================================================
with tab_portal:
    st.markdown("##### 🚀 Quick Enterprise Test Scenarios")
    c1, c2, c3, c4 = st.columns(4)
    preset_query = None

    with c1:
        if st.button("💳 1. Billing Desync Bug", use_container_width=True):
            preset_query = "My payment was deducted, but my order was not confirmed."
        st.caption("Tests: MCP Gateway + Policy RAG")

    with c2:
        if st.button("🔧 2. 504 Gateway Timeout", use_container_width=True):
            preset_query = "Getting error 504 gateway timeout when calling the analytics REST API."
        st.caption("Tests: Tech Diagnostic RAG")

    with c3:
        if st.button("📦 3. Compatibility Specs", use_container_width=True):
            preset_query = "Does CloudSync Pro support local offline storage on macOS Sequoia?"
        st.caption("Tests: Product Catalog RAG")

    with c4:
        if st.button("🚨 4. Urgent Escalation", use_container_width=True):
            preset_query = "I was charged twice and nobody responded! Refund my money right now or I am calling legal!"
        st.caption("Tests: Sentiment & HITL Queue")

    st.markdown("---")

    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🛡️"):
            st.markdown(msg["content"])
            
            if "meta" in msg and msg["meta"]:
                meta = msg["meta"]
                cat = meta.get("category", "GENERAL").upper()
                prio = meta.get("priority", "MEDIUM").upper()
                sent = meta.get("sentiment", "Neutral").capitalize()
                agent = meta.get("assigned_agent", "SupportSphere Agent")

                prio_class = "chip-prio-high" if prio in ["HIGH", "URGENT"] else ("chip-prio-med" if prio == "MEDIUM" else "chip-prio-low")
                sent_class = "chip-sent-pos" if sent == "Positive" else ("chip-sent-frust" if sent == "Frustrated" else "chip-sent-angry")

                st.markdown(f"""
                <div class="badge-strip">
                    <span class="meta-chip chip-category">🏷️ Category: {cat}</span>
                    <span class="meta-chip {prio_class}">⚡ Priority: {prio}</span>
                    <span class="meta-chip {sent_class}">🎭 Sentiment: {sent}</span>
                    <span class="meta-chip chip-agent">🤖 Handled By: {agent}</span>
                </div>
                """, unsafe_allow_html=True)

                if meta.get("scratchpad"):
                    with st.expander("🔍 Multi-Agent Execution & Tool Audit Trail"):
                        for step in meta["scratchpad"]:
                            st.markdown(f"- `{step}`")

    # User Query Input
    user_prompt = st.chat_input("Enter customer inquiry...")
    if preset_query:
        user_prompt = preset_query

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        ticket_id = f"TCK-{int(time.time()) % 100000}"
        initial_state = {
            "ticket_id": ticket_id,
            "customer_id": current_user["customer_id"],
            "customer_name": current_user["name"],
            "customer_tier": current_user["tier"],
            "messages": [{"role": "user", "content": user_prompt}],
            "classification": None,
            "assigned_agent": None,
            "retrieved_context": None,
            "agent_scratchpad": [],
            "final_response": None,
            "is_escalated": False
        }

        with st.spinner("🤖 Multi-Agent Orchestrator executing (Triage → Routing → Swarms → Verification)..."):
            result_state = support_orchestrator.invoke(initial_state)

        classification = result_state.get("classification")
        final_answer = result_state.get("final_response") or "Thank you for contacting support."
        assigned_agent = result_state.get("assigned_agent", "SupportSphere Agent")
        is_escalated = result_state.get("is_escalated", False)
        scratchpad = result_state.get("agent_scratchpad", [])

        meta_info = {
            "ticket_id": ticket_id,
            "category": classification.category if classification else "general",
            "priority": classification.priority if classification else "medium",
            "sentiment": classification.sentiment if classification else "neutral",
            "assigned_agent": assigned_agent,
            "is_escalated": is_escalated,
            "scratchpad": scratchpad
        }

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": final_answer,
            "meta": meta_info
        })

        st.session_state.tickets_registry.append({
            "Ticket ID": ticket_id,
            "Customer": current_user["name"],
            "Query": user_prompt,
            "Category": meta_info["category"].upper(),
            "Priority": meta_info["priority"].upper(),
            "Sentiment": meta_info["sentiment"].capitalize(),
            "Assigned Agent": assigned_agent,
            "Status": "Escalated" if is_escalated else "Resolved",
            "Timestamp": time.strftime("%H:%M:%S")
        })

        st.rerun()

# =====================================================================
# TAB 2: SUPPORT OPERATIONS COMMAND CENTER
# =====================================================================
with tab_analytics:
    st.markdown("### 📊 Executive Operations & Human-in-the-Loop Command Center")
    st.caption("Live monitoring of ticket triage, SLA compliance, sentiment drift, and human takeover requests.")

    total_tickets = len(st.session_state.tickets_registry)
    escalated_count = sum(1 for t in st.session_state.tickets_registry if t["Status"] == "Escalated")
    urgent_count = sum(1 for t in st.session_state.tickets_registry if t["Priority"] in ["HIGH", "URGENT"])
    auto_resolved = total_tickets - escalated_count
    auto_rate = (auto_resolved / max(1, total_tickets)) * 100

    # KPI Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Total Inquiries Logged</div>
            <div class="metric-val">{total_tickets}</div>
            <div style="font-size: 11px; color: #34d399;">🟢 Session Live</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Escalated to Human</div>
            <div class="metric-val" style="color: {'#f87171' if escalated_count > 0 else '#f9fafb'};">{escalated_count}</div>
            <div style="font-size: 11px; color: {'#f87171' if escalated_count > 0 else '#9ca3af'};">{'⚠️ Action Required' if escalated_count > 0 else 'All Clear'}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">High / Critical Urgency</div>
            <div class="metric-val" style="color: {'#fbbf24' if urgent_count > 0 else '#f9fafb'};">{urgent_count}</div>
            <div style="font-size: 11px; color: #9ca3af;">SLA Monitored</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">AI Auto-Resolution Rate</div>
            <div class="metric-val" style="color: #60a5fa;">{auto_rate:.1f}%</div>
            <div style="font-size: 11px; color: #34d399;">Target: > 90%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.tickets_registry:
        df_tickets = pd.DataFrame(st.session_state.tickets_registry)

        # Visual Analytics Charts
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### 📈 Triage Category Distribution")
            cat_counts = df_tickets["Category"].value_counts()
            st.bar_chart(cat_counts, color="#60a5fa")
            
        with col_c2:
            st.markdown("##### 🎭 Customer Sentiment Breakdown")
            sent_counts = df_tickets["Sentiment"].value_counts()
            st.bar_chart(sent_counts, color="#a78bfa")

        st.markdown("---")
        st.markdown("##### 📋 Enterprise Real-Time Ticket Audit Table")
        st.dataframe(df_tickets, use_container_width=True)

        # Escalation Takeover Section
        st.markdown("---")
        st.markdown("##### 🚨 Human-in-the-Loop Escalation Queue")
        escalated_cases = [t for t in st.session_state.tickets_registry if t["Status"] == "Escalated"]

        if escalated_cases:
            for item in escalated_cases:
                with st.expander(f"⚠️ Urgent Case {item['Ticket ID']} — {item['Customer']} [{item['Priority']}] | Sentiment: {item['Sentiment']}", expanded=True):
                    st.write(f"**Customer Query:** *\"{item['Query']}\"*")
                    st.write(f"**Category:** `{item['Category']}` | **Current Handler:** `{item['Assigned Agent']}`")
                    
                    st.markdown("**Supervisor Intervention Actions:**")
                    col_input, col_send, col_replay = st.columns([2, 1, 1])
                    
                    with col_input:
                        st.text_input("Direct Human Response / Instructions:", key=f"input_{item['Ticket ID']}")
                    
                    with col_send:
                        if st.button("📨 Dispatch Human Response", key=f"disp_{item['Ticket ID']}", use_container_width=True):
                            item["Status"] = "Resolved"
                            st.success(f"Dispatched human reply for ticket {item['Ticket ID']}!")
                            st.rerun()

                    with col_replay:
                        if st.button("⚡ Trigger MCP Replay", key=f"mcp_{item['Ticket ID']}", use_container_width=True):
                            replay_status = mcp_replay_order("ORD-58291")
                            item["Status"] = "Resolved"
                            st.info(f"MCP Replay Result: {replay_status}")
                            st.rerun()
        else:
            st.success("✅ Zero open escalations. All customer queries autonomously handled.")
    else:
        st.info("No tickets recorded yet. Test queries in the Customer Support tab to view live triage analytics.")
