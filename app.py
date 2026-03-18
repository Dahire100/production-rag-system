import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings; warnings.filterwarnings("ignore")

import pickle, datetime, streamlit as st
from src.auth          import is_logged_in, get_current_user, render_login_page, logout
from src.audit         import log_query, log_feedback, read_audit_log, read_feedback_log
from src.conversations import (new_conversation, save_conversation, load_conversation,
                                list_conversations, delete_conversation)
from src.ingest        import ingest_uploaded_file, is_already_ingested, SUPPORTED_EXTENSIONS
from src.generate      import (stream_answer, get_last_docs, get_followup_suggestions,
                                summarize_document, reset_chain)
from src.config        import (CHUNKS_PKL, GROQ_MODEL, EMBED_MODEL, RERANK_MODEL,
                                CHUNK_SIZE, RERANK_TOP_N, MULTI_QUERY_N)
from src.observability import (get_store, get_latency_percentiles,
                                get_cost_summary, get_quality_summary, get_user_usage)
from dotenv import load_dotenv
load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Enterprise AI Assistant", page_icon="💼",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu,header,footer{visibility:hidden}
    [data-testid="stSidebar"]{
        background:linear-gradient(180deg,#0a0a16 0%,#0f0f24 60%,#0a1428 100%);
        border-right:1px solid rgba(255,255,255,.06)
    }
    [data-testid="stSidebar"] *{color:#d0d0e0!important}
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3{color:#fff!important}
    [data-testid="stFileUploader"]{border:2px dashed rgba(100,150,255,.3);
        border-radius:12px;padding:10px;background:rgba(100,150,255,.04)}
    .stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid rgba(255,255,255,.08)}
    .stTabs [data-baseweb="tab"]{border-radius:8px 8px 0 0;padding:8px 20px;font-weight:600}
    .pill{display:inline-block;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600;margin:2px 0}
    .pill-ok  {background:rgba(46,204,113,.15);color:#2ecc71!important}
    .pill-warn{background:rgba(231,76,60,.15) ;color:#e74c3c!important}
    .pill-info{background:rgba(100,150,255,.15);color:#aac4ff!important}
    .pill-neu {background:rgba(255,255,255,.07);color:#aaa!important}
    .doc-item{background:rgba(255,255,255,.04);border-radius:8px;
              padding:7px 10px;margin:4px 0;font-size:13px}
    .conv-item{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);
               border-radius:8px;padding:8px 12px;margin:4px 0;cursor:pointer;font-size:13px}
    .conv-item:hover{background:rgba(100,150,255,.08)}
    .conv-active{border-color:rgba(100,150,255,.5)!important;background:rgba(100,150,255,.1)!important}
    .cit-card{background:rgba(52,152,219,.07);border-left:3px solid #3498db;
              border-radius:8px;padding:10px 14px;margin:6px 0;font-size:13px;line-height:1.55}
    .cit-head{font-weight:700;color:#74b9ff;font-size:12px;margin-bottom:5px}
    .chunk-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
                border-radius:10px;padding:12px 16px;margin:8px 0;font-size:13px;line-height:1.6}
    .chunk-meta{color:#888;font-size:11px;margin-bottom:6px}
    .summary-box{background:rgba(46,213,115,.06);border-left:3px solid #2ecc71;
                 border-radius:8px;padding:14px 18px;margin:10px 0;font-size:14px;line-height:1.65}
    .stat-card{background:rgba(255,255,255,.04);border-radius:12px;
               padding:16px;text-align:center;border:1px solid rgba(255,255,255,.07)}
    .stat-num{font-size:28px;font-weight:800;color:#aac4ff}
    .stat-lbl{font-size:12px;color:#888;margin-top:2px}

    /* Grafana styles */
    .grafana-panel {
        background-color: #111217;
        border: 1px solid #2b2d39;
        border-radius: 4px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .grafana-title {
        color: #B6C0CF;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .grafana-value {
        font-size: 32px;
        font-weight: 700;
        line-height: 1.2;
    }
    .grafana-value.green { color: #73BF69; }
    .grafana-value.red { color: #F2495C; }
    .grafana-value.orange { color: #FF9830; }
    .grafana-value.blue { color: #5794F2; }
    .grafana-subtext {
        font-size: 11px;
        color: #8E9BAE;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Auth gate ─────────────────────────────────────────────────────────────────
if not is_logged_in():
    render_login_page()
    st.stop()

user = get_current_user()

# ── Session init ──────────────────────────────────────────────────────────────
def _ss(key, default):
    if key not in st.session_state: st.session_state[key] = default

_ss("messages",       [])
_ss("followups",      [])
_ss("pending_prompt", None)
_ss("query_count",    0)
_ss("custom_prompt",  "")
_ss("active_conv_id", None)
_ss("pending_fb",     None)   # {question, answer} awaiting feedback

# Load or create a conversation on first login
if st.session_state.active_conv_id is None:
    convs = list_conversations(user)
    if convs:
        latest = load_conversation(user, convs[0]["id"])
        st.session_state.active_conv_id = convs[0]["id"]
        st.session_state.messages       = latest.get("messages", [])
    else:
        cid = new_conversation(user)
        st.session_state.active_conv_id = cid
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                f"👋 **Welcome, {user}!**\n\n"
                "I'm your Enterprise AI Assistant. Upload a document, then ask me anything.\n\n"
                "I use **Hybrid RAG + Multi-Query expansion** for maximum accuracy, "
                "with every answer strictly cited back to your source."
            )
        }]

# ── Index helper ──────────────────────────────────────────────────────────────
chunks_exist  = os.path.exists(CHUNKS_PKL)
doc_chunks_map = {}
if chunks_exist:
    try:
        with open(CHUNKS_PKL,"rb") as f: all_chunks = pickle.load(f)
        for c in all_chunks:
            fn = os.path.basename(c.metadata.get("source","unknown"))
            doc_chunks_map.setdefault(fn,[]).append(c)
    except Exception: pass
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # User badge
    if user == "admin":
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">'
            f'<div style="width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#667eea,#764ba2);'
            f'display:flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;color:#fff">'
            f'{user[0].upper()}</div>'
            f'<div><div style="font-weight:700;font-size:15px">Administrator</div>'
            f'<div style="font-size:11px;color:#666">Unlimited Access</div></div></div>',
            unsafe_allow_html=True)
    else:
        usage = get_user_usage(user)
        tokens_used = usage["tokens"]
        cost_usd = usage["cost_usd"]
        limit_pct = min(100, int(tokens_used / 1000 * 100))
        bg_color = "#e74c3c" if limit_pct >= 100 else "#2ecc71"
        
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
            f'<div style="width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#667eea,#764ba2);'
            f'display:flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;color:#fff">'
            f'{user[0].upper()}</div>'
            f'<div><div style="font-weight:700;font-size:15px">{user}</div>'
            f'<div style="font-size:11px;color:#aac4ff">Cost to date: ${cost_usd:.5f}</div></div></div>'
            f'<div style="font-size:11px;color:#888;margin-bottom:4px;display:flex;justify-content:space-between">'
            f'<span>Free Tokens: <b>{tokens_used:,}</b> / 1,000</span>'
            f'<span>{limit_pct}%</span></div>'
            f'<div style="background:rgba(255,255,255,.1);height:4px;border-radius:2px;margin-bottom:12px">'
            f'<div style="background:{bg_color};height:100%;width:{limit_pct}%;border-radius:2px"></div></div>',
            unsafe_allow_html=True)
        
    if st.button("Sign Out", use_container_width=True): logout(); st.rerun()

    st.markdown("---")

    # ── Conversations ─────────────────────────────────────────────────────────
    st.markdown("#### 💬 Conversations")
    if st.button("＋ New Chat", use_container_width=True):
        cid = new_conversation(user)
        st.session_state.active_conv_id = cid
        st.session_state.messages       = []
        st.session_state.followups      = []
        st.rerun()

    for conv in list_conversations(user)[:8]:
        active_cls = "conv-active" if conv["id"] == st.session_state.active_conv_id else ""
        col_t, col_d = st.columns([4,1])
        with col_t:
            if st.button(
                f"{'🟢 ' if active_cls else '💬 '}{conv['title'][:26]}",
                key=f"conv_{conv['id']}",
                use_container_width=True
            ):
                data = load_conversation(user, conv["id"])
                st.session_state.active_conv_id = conv["id"]
                st.session_state.messages       = data.get("messages",[])
                st.session_state.followups      = []
                st.rerun()
        with col_d:
            if st.button("🗑", key=f"del_{conv['id']}"):
                delete_conversation(user, conv["id"])
                if conv["id"] == st.session_state.active_conv_id:
                    st.session_state.active_conv_id = None
                    st.session_state.messages = []
                st.rerun()

    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────────────────────
    exts = ", ".join(e.lstrip(".").upper() for e in SUPPORTED_EXTENSIONS)
    st.markdown(f"#### 📄 Upload Documents\n<span style='font-size:11px;color:#555'>Supported: {exts}</span>",
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop file", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
                                label_visibility="collapsed")
    if uploaded:
        fpath = os.path.join("data", uploaded.name)
        os.makedirs("data", exist_ok=True)
        with open(fpath,"wb") as f: f.write(uploaded.getbuffer())
        if is_already_ingested(fpath):
            st.info(f"ℹ️ **{uploaded.name}** already indexed.")
        else:
            prog = st.progress(0, text="Preparing…")
            try:
                prog.progress(20, text="Reading file…")
                n = ingest_uploaded_file(fpath)
                prog.progress(100, text="Done!")
                st.success(f"✅ Indexed **{n}** segments")
                prog.empty()
                st.rerun()
            except Exception as e:
                prog.empty(); st.error(f"❌ {e}")

    # ── Doc list ──────────────────────────────────────────────────────────────
    if doc_chunks_map:
        st.markdown("#### 📋 Knowledge Base")
        for fname, clist in doc_chunks_map.items():
            st.markdown(
                f'<div class="doc-item">📄 {fname}'
                f'<span style="float:right;color:#555;font-size:11px">{len(clist)} seg</span></div>',
                unsafe_allow_html=True)
        if st.button("🗑️ Clear All", use_container_width=True):
            reset_chain()
            for f in [CHUNKS_PKL,"bm25_index.pkl","ingested_hashes.pkl"]:
                try:
                    if os.path.exists(f): os.remove(f)
                except Exception: pass
            import shutil
            if os.path.exists("db"): shutil.rmtree("db", ignore_errors=True)
            st.rerun()

    st.markdown("---")

    # ── Status ────────────────────────────────────────────────────────────────
    st.markdown("#### System Status")
    def pill(label, ok):
        c = "pill-ok" if ok else "pill-warn"
        return f'<div class="pill {c}">{"✓" if ok else "✗"} {label}</div>'
    st.markdown(pill("Groq API",     bool(os.getenv("GROQ_API_KEY"))),   unsafe_allow_html=True)
    st.markdown(pill("Cohere API",   bool(os.getenv("COHERE_API_KEY"))), unsafe_allow_html=True)
    st.markdown(pill("Index Ready",  chunks_exist),                       unsafe_allow_html=True)
    total_segs = sum(len(v) for v in doc_chunks_map.values())
    if total_segs:
        st.markdown(f'<div class="pill pill-info">📊 {total_segs} segments</div>', unsafe_allow_html=True)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_chat, tab_explorer, tab_admin, tab_obs, tab_settings = st.tabs(
    ["💬 Chat", "🔍 Document Explorer", "📊 Admin Panel", "🔭 Observability", "⚙️ Settings"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    hdr, col_exp, col_clr = st.columns([6,1,1])
    with hdr: st.markdown("### 💬 Chat")
    with col_exp:
        lines = [f"# Chat Export — {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n---\n"]
        for m in st.session_state.messages:
            lines.append(f"**{'You' if m['role']=='user' else 'Assistant'}:**\n{m['content']}\n")
        st.download_button("⬇️ Export", "\n".join(lines), "chat_export.md",
                           "text/markdown", use_container_width=True)
    with col_clr:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages    = []
            st.session_state.followups   = []
            st.session_state.query_count = 0
            if st.session_state.active_conv_id:
                save_conversation(user, st.session_state.active_conv_id, [])
            st.rerun()

    # Stats bar
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Queries",    st.session_state.query_count)
    c2.metric("Documents",  len(doc_chunks_map))
    c3.metric("Segments",   total_segs)
    c4.metric("Session",    user)

    st.divider()

    # Render messages
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("contexts"):
                with st.expander(f"📚 {len(msg['contexts'])} Citations", expanded=False):
                    for i,ctx in enumerate(msg["contexts"]):
                        st.markdown(
                            f'<div class="cit-card">'
                            f'<div class="cit-head">📄 [{msg["sources"][i]}] — '
                            f'{msg.get("filenames",[""])[i]} · Page {msg.get("pages",["?"])[i]}</div>'
                            f'{ctx[:600]}{"…" if len(ctx)>600 else ""}'
                            f'</div>', unsafe_allow_html=True)

            # Feedback buttons (only on last assistant message)
            if (msg["role"] == "assistant" and msg.get("contexts")
                    and idx == len(st.session_state.messages) - 1):
                fb_col1, fb_col2, _ = st.columns([1,1,8])
                if fb_col1.button("👍", key=f"thumbsup_{idx}"):
                    log_feedback(user, st.session_state.messages[idx-1]["content"],
                                 msg["content"], "positive")
                    st.toast("Thanks for the feedback! 🎉")
                if fb_col2.button("👎", key=f"thumbsdn_{idx}"):
                    log_feedback(user, st.session_state.messages[idx-1]["content"],
                                 msg["content"], "negative")
                    st.toast("Noted — we'll work to improve!")

    # Follow-up chips
    if st.session_state.followups:
        st.markdown("**💡 Suggested follow-ups:**")
        cols = st.columns(len(st.session_state.followups))
        for i,s in enumerate(st.session_state.followups):
            if cols[i].button(s, key=f"fu_{i}", use_container_width=True):
                st.session_state.pending_prompt = s
                st.session_state.followups = []
                st.rerun()

    # Pending feedback comment (inline)
    if st.session_state.pending_fb:
        with st.form("fb_form", clear_on_submit=True):
            comment = st.text_input("Optional comment (what was wrong?)")
            if st.form_submit_button("Submit Feedback"):
                log_feedback(user, st.session_state.pending_fb["question"],
                             st.session_state.pending_fb["answer"], "negative", comment)
                st.session_state.pending_fb = None
                st.toast("Feedback submitted. Thank you!")

    # Chat input
    typed = st.chat_input("Ask anything about your documents…")
    active = st.session_state.pending_prompt or typed
    if st.session_state.pending_prompt: st.session_state.pending_prompt = None

    if active:
        # Check token limits!
        current_usage = get_user_usage(user)
        if user != "admin" and current_usage["tokens"] >= 1000:
            with st.chat_message("assistant"):
                st.error("🚫 **Free tier limit reached.** You have used all 1,000 free tokens. Please upgrade your plan or contact your administrator.")
            st.stop()

        st.session_state.messages.append({"role":"user","content":active})
        with st.chat_message("user"): st.markdown(active)

        with st.chat_message("assistant"):
            history = st.session_state.messages[:-1]
            full_ans = st.write_stream(
                stream_answer(active, history=history,
                              extra_instructions=st.session_state.custom_prompt,
                              user=user)
            )
            docs     = get_last_docs()
            contexts = [d.page_content for d in docs]
            sources  = [d.metadata.get("doc_id","?") for d in docs]
            fnames   = [os.path.basename(d.metadata.get("source","")) for d in docs]
            pages    = [d.metadata.get("page","?") for d in docs]

            if contexts:
                with st.expander(f"📚 {len(contexts)} Citations", expanded=False):
                    for i,ctx in enumerate(contexts):
                        st.markdown(
                            f'<div class="cit-card">'
                            f'<div class="cit-head">📄 [{sources[i]}] — {fnames[i]} · Page {pages[i]}</div>'
                            f'{ctx[:600]}{"…" if len(ctx)>600 else ""}'
                            f'</div>', unsafe_allow_html=True)

            st.session_state.messages.append({
                "role":"assistant","content":full_ans,
                "contexts":contexts,"sources":sources,"filenames":fnames,"pages":pages,
            })
            st.session_state.query_count += 1
            log_query(user, active, full_ans, len(sources),
                      str(st.session_state.active_conv_id))
            st.session_state.followups = get_followup_suggestions(active, full_ans[:800])
            save_conversation(user, st.session_state.active_conv_id,
                              st.session_state.messages)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_explorer:
    st.markdown("### 🔍 Document Explorer")
    if not doc_chunks_map:
        st.info("No documents indexed yet. Upload a file in the sidebar.")
    else:
        left, right = st.columns([1, 2])
        with left:
            st.markdown("#### Select Document")
            sel_doc = st.selectbox("Doc", list(doc_chunks_map.keys()),
                                   label_visibility="collapsed")
            segs    = doc_chunks_map.get(sel_doc, [])
            fmt     = segs[0].metadata.get("format","?").upper() if segs else "?"
            pg_max  = max((c.metadata.get("page",1) for c in segs), default=1)
            st.markdown(f"**Format:** {fmt} · **{len(segs)}** segments · **{pg_max}** pages")

            if st.button("🧠 Summarize Document", use_container_width=True, type="primary"):
                with st.spinner("Generating summary…"):
                    summary = summarize_document(sel_doc, segs)
                    st.session_state[f"sum_{sel_doc}"] = summary

        with right:
            if f"sum_{sel_doc}" in st.session_state:
                st.markdown("#### 📝 AI Summary")
                st.markdown(
                    f'<div class="summary-box">{st.session_state[f"sum_{sel_doc}"]}</div>',
                    unsafe_allow_html=True)
                if st.button("✕ Close Summary"): del st.session_state[f"sum_{sel_doc}"]; st.rerun()

        st.divider()
        st.markdown("#### 🔎 Search Segments")
        sq = st.text_input("Keyword search within document",
                           placeholder="e.g. variables, loops, recursion…",
                           label_visibility="collapsed")
        page_filter = st.slider("Filter by page", 1, int(pg_max), (1, int(pg_max))) if pg_max > 1 else None

        display_segs = segs
        if page_filter:
            display_segs = [c for c in display_segs
                            if page_filter[0] <= c.metadata.get("page",1) <= page_filter[1]]
        if sq:
            display_segs = [c for c in display_segs if sq.lower() in c.page_content.lower()]

        st.caption(f"Showing **{min(20,len(display_segs))}** of **{len(display_segs)}** segments")
        for chunk in display_segs[:20]:
            page = chunk.metadata.get("page","?")
            did  = chunk.metadata.get("doc_id","?")
            text = chunk.page_content
            hi   = text.replace(sq, f"**{sq}**") if sq else text
            st.markdown(
                f'<div class="chunk-card">'
                f'<div class="chunk-meta">📄 [{did}] · Page {page}</div>'
                f'{hi[:500]}{"…" if len(text)>500 else ""}'
                f'</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ADMIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_admin:
    # Only admin user sees full audit log
    is_admin = (user == "admin")
    st.markdown("### 📊 Admin Panel")
    if not is_admin:
        st.warning("🔒 Full analytics are visible to admin users only.")

    audit    = read_audit_log()
    feedback = read_feedback_log()

    # KPI cards
    k1,k2,k3,k4 = st.columns(4)
    pos = sum(1 for r in feedback if r.get("rating")=="positive")
    neg = sum(1 for r in feedback if r.get("rating")=="negative")
    sat = round(100*pos/(pos+neg),1) if (pos+neg) else 0
    k1.markdown(f'<div class="stat-card"><div class="stat-num">{len(audit)}</div><div class="stat-lbl">Total Queries</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="stat-card"><div class="stat-num">{len(feedback)}</div><div class="stat-lbl">Feedback Items</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="stat-card"><div class="stat-num">{sat}%</div><div class="stat-lbl">Satisfaction</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="stat-card"><div class="stat-num">{len(doc_chunks_map)}</div><div class="stat-lbl">Documents</div></div>', unsafe_allow_html=True)

    st.divider()

    if is_admin and audit:
        import pandas as pd
        st.markdown("#### 🗒️ Recent Query Audit Log")
        df_audit = pd.DataFrame(audit[-50:][::-1])
        st.dataframe(df_audit[["timestamp","user","question","num_sources"]], use_container_width=True)

        # Download
        st.download_button("⬇️ Download Full Audit CSV",
                           open("audit.csv","rb").read() if os.path.exists("audit.csv") else b"",
                           "audit.csv","text/csv")

    st.divider()

    if is_admin:
        st.markdown(
            """
            <div style="padding:16px; border-radius:12px; background:linear-gradient(90deg, #111217, #1c1e26); border-left:4px solid #5794F2; margin-bottom:16px;">
                <h4 style="margin:0; color:#fff; font-size:18px;">🕵️‍♂️ User Chat Explorer</h4>
                <p style="margin:4px 0 0 0; color:#B6C0CF; font-size:13px;">Securely inspect localized conversation histories across all user accounts.</p>
            </div>
            """, unsafe_allow_html=True
        )
        
        from src.config import CONVERSATIONS_DIR
        import os, json
        
        if os.path.exists(CONVERSATIONS_DIR):
            users = [u for u in os.listdir(CONVERSATIONS_DIR) if os.path.isdir(os.path.join(CONVERSATIONS_DIR, u))]
            if users:
                c1, c2 = st.columns(2)
                with c1:
                    user_sel = st.selectbox("Select Target User", users)
                
                user_convs = list_conversations(user_sel)
                if user_convs:
                    with c2:
                        conv_id = st.selectbox("Select Chat History", [c["id"] for c in user_convs], 
                                               format_func=lambda x: next((f"{c['title']} ({c['count']} msgs)" for c in user_convs if c["id"] == x), x))
                    
                    if conv_id:
                        chat_data = load_conversation(user_sel, conv_id)
                        if chat_data and "messages" in chat_data:
                            # Render chat in an elegant container box
                            st.markdown(f"##### Transcript: `{conv_id}`")
                            with st.container(border=True, height=500):
                                for m in chat_data["messages"]:
                                    # Use native chat_message components to make it look like a real conversation!
                                    with st.chat_message(m["role"]):
                                        st.markdown(m["content"])
                                        # Render citations if they exist in history
                                        if m.get("contexts"):
                                            with st.expander(f"📚 {len(m['contexts'])} Citations", expanded=False):
                                                for i, ctx in enumerate(m["contexts"]):
                                                    st.markdown(
                                                        f'<div class="cit-card">'
                                                        f'<div class="cit-head">📄 [{m.get("sources", ["?"])[i]}]'
                                                        f'</div>{ctx[:200]}…</div>', unsafe_allow_html=True)
                else:
                    st.info(f"{user_sel} currently has no stored conversations.")
            else:
                st.info("No registered users with conversation histories found.")
        st.divider()

    if is_admin and feedback:
        import pandas as pd
        st.markdown("#### 👍👎 Feedback Log")
        df_fb = pd.DataFrame(feedback[-50:][::-1])
        pos_df = df_fb[df_fb["rating"]=="positive"]
        neg_df = df_fb[df_fb["rating"]=="negative"]
        fc1,fc2 = st.columns(2)
        fc1.metric("Positive", len(pos_df))
        fc2.metric("Negative", len(neg_df))
        st.dataframe(df_fb[["timestamp","user","question","rating","comment"]],
                     use_container_width=True)
        st.download_button("⬇️ Download Feedback CSV",
                           open("feedback.csv","rb").read() if os.path.exists("feedback.csv") else b"",
                           "feedback.csv","text/csv")

    if not audit and not feedback:
        st.info("No data yet — queries and feedback will appear here as users interact.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — OBSERVABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_obs:
    import time as _time
    import math

    # Header and controls
    obs_col1, obs_col2 = st.columns([3, 1])
    with obs_col1:
        st.markdown("<h2 style='margin-bottom:0;color:#fff'>Grafana Observability</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#B6C0CF;font-size:14px;margin-top:0'>Real-time telemetry & RAG performance metrics</p>", unsafe_allow_html=True)
    with obs_col2:
        window_h = st.selectbox(
            "Time Range",
            options=[1, 6, 12, 24, 48, 168],
            index=3,
            format_func=lambda h: f"Last {h} hours" if h < 168 else "Last 7 days",
            label_visibility="collapsed"
        )

    is_admin_obs = (user == "admin")
    if not is_admin_obs:
        st.warning("🔒 Full observability metrics are visible to admin users only.")

    obs_store = get_store()

    # ── ROW 1: Latency Gauges ──────────────────────────────────────────
    st.markdown("<h4 style='color:#E0E6ED;border-bottom:1px solid #2b2d39;padding-bottom:8px;margin-top:20px'>Latency Percentiles</h4>", unsafe_allow_html=True)
    ops = ["rag_e2e", "retrieval", "llm_generate", "summarize"]
    op_labels = {
        "rag_e2e":      "End-to-End",
        "retrieval":    "Retrieval",
        "llm_generate": "LLM Generate",
        "summarize":    "Summarize",
    }
    lat_cols = st.columns(len(ops))
    
    for i, op in enumerate(ops):
        pct = get_latency_percentiles(op, window_hours=window_h)
        # Determine color based on p50
        val = pct["p50"]
        if val < 2.0: color_cls = "green"
        elif val < 5.0: color_cls = "orange"
        else: color_cls = "red"
        
        with lat_cols[i]:
            st.markdown(
                f'<div class="grafana-panel">'
                f'<div class="grafana-title">{op_labels[op]} P50</div>'
                f'<div class="grafana-value {color_cls}">{val} <span style="font-size:16px">s</span></div>'
                f'<div class="grafana-subtext">P95: {pct["p95"]}s | n={pct["count"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── ROW 2: Quality Metrics (RAGAS) ─────────────────────────────────
    st.markdown("<h4 style='color:#E0E6ED;border-bottom:1px solid #2b2d39;padding-bottom:8px;margin-top:10px'>Quality Gate (CI)</h4>", unsafe_allow_html=True)
    q_summary = get_quality_summary(window_hours=window_h)

    if q_summary["count"] == 0:
        st.info("No quality scores recorded yet. Run `python eval/regression_gate.py` to populate.")
    else:
        from src.observability import QUALITY_THRESHOLDS
        qm = [
            ("Faithfulness",      q_summary["faithfulness"],      QUALITY_THRESHOLDS["faithfulness"]),
            ("Answer Relevancy",  q_summary["answer_relevancy"],  QUALITY_THRESHOLDS["answer_relevancy"]),
            ("Context Precision", q_summary["context_precision"], QUALITY_THRESHOLDS["context_precision"]),
        ]
        q_cols = st.columns(len(qm))
        for i, (label, value, threshold) in enumerate(qm):
            ok = value >= threshold
            color_cls = "green" if ok else "red"
            with q_cols[i]:
                st.markdown(
                    f'<div class="grafana-panel">'
                    f'<div class="grafana-title">{label}</div>'
                    f'<div class="grafana-value {color_cls}">{value:.3f}</div>'
                    f'<div class="grafana-subtext">Threshold: {threshold} ({"PASS" if ok else "FAIL"})</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── ROW 3: Cost & Usage Dashboard ──────────────────────────────────
    cost = get_cost_summary(window_hours=window_h)
    
    st.markdown("<h4 style='color:#E0E6ED;border-bottom:1px solid #2b2d39;padding-bottom:8px;margin-top:10px'>Usage & Infrastructure</h4>", unsafe_allow_html=True)
    
    cc1, cc2, cc3 = st.columns([1,1,2])
    with cc1:
        st.markdown(
            f'<div class="grafana-panel" style="align-items:flex-start">'
            f'<div class="grafana-title">Total Spend</div>'
            f'<div class="grafana-value blue">${cost["total_cost_usd"]:.5f}</div>'
            f'<div class="grafana-subtext">{cost["total_requests"]} LLM Requests</div>'
            f'</div>', unsafe_allow_html=True)
            
    with cc2:
        st.markdown(
            f'<div class="grafana-panel" style="align-items:flex-start">'
            f'<div class="grafana-title">Token Volume</div>'
            f'<div class="grafana-value blue">{cost["total_input_tokens"] + cost["total_output_tokens"]:,}</div>'
            f'<div class="grafana-subtext">In: {cost["total_input_tokens"]:,} | Out: {cost["total_output_tokens"]:,}</div>'
            f'</div>', unsafe_allow_html=True)
            
    with cc3:
        if is_admin_obs:
            ts_data = obs_store.get_cost_timeseries(window_hours=window_h)
            if len(ts_data) >= 2:
                import pandas as pd
                df_ts = pd.DataFrame(ts_data)
                df_ts["time"] = pd.to_datetime(df_ts["bucket_ts"], unit="s")
                df_ts = df_ts.set_index("time")
                
                # Make it look a bit more like Grafana charts by adjusting theme
                st.area_chart(df_ts["cost"], height=130, use_container_width=True, color="#5794F2")
            else:
                st.markdown(
                    f'<div class="grafana-panel" style="height:130px;justify-content:center">'
                    f'<div class="grafana-subtext">Not enough data points for timeseries chart</div>'
                    f'</div>', unsafe_allow_html=True)


    # ── ROW 4: Distributed Traces  ─────────────────────────────────────
    if is_admin_obs:
        st.markdown("<h4 style='color:#E0E6ED;border-bottom:1px solid #2b2d39;padding-bottom:8px;margin-top:10px'>Traces Explorer</h4>", unsafe_allow_html=True)
        traces = obs_store.get_recent_traces(limit=25)
        if traces:
            import pandas as pd
            df_tr = pd.DataFrame(traces)
            
            display_tr = [c for c in ["name","status","latency_s","trace_cost","input_tokens","output_tokens","trace_id"]
                          if c in df_tr.columns]
                          
            # Format dataframe to look a bit cleaner
            df_tr["latency_s"] = df_tr["latency_s"].apply(lambda x: f"{x:.3f} s" if x else "—")
            if "trace_cost" in df_tr:
                df_tr["trace_cost"] = df_tr["trace_cost"].apply(lambda x: f"${x:.6f}" if x else "$0.000000")
            
            st.dataframe(
                df_tr[display_tr].rename(columns={
                    "name": "Operation", "status": "Status", "latency_s": "Latency",
                    "trace_cost": "Cost", "input_tokens": "Tokens (In)",
                    "output_tokens": "Tokens (Out)", "trace_id": "Trace ID"
                }),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No traces recorded yet.")

        colA, colB = st.columns([1,5])
        with colA:
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_settings:
    st.markdown("### ⚙️ Settings")
    s1,s2 = st.columns(2)

    with s1:
        st.markdown("#### 🧑‍💼 Assistant Persona")
        st.caption("Append custom instructions to every answer — shape tone, format, and focus.")
        custom = st.text_area("Instructions", value=st.session_state.custom_prompt,
                              placeholder="e.g. 'Reply only in bullet points.' 'Use beginner-friendly language.'",
                              height=130, label_visibility="collapsed")
        if st.button("💾 Save", use_container_width=True, type="primary"):
            st.session_state.custom_prompt = custom
            st.success("✅ Saved! Applied to your next question.")
        if st.session_state.custom_prompt:
            st.info(f"**Active:** {st.session_state.custom_prompt[:140]}")
            if st.button("✕ Clear"): st.session_state.custom_prompt = ""; st.rerun()

        st.markdown("#### 🗂️ Conversation")
        if st.session_state.active_conv_id:
            new_title = st.text_input("Rename this chat",
                                      placeholder="e.g. Python Notes Q&A")
            if st.button("✎ Rename", use_container_width=True):
                save_conversation(user, st.session_state.active_conv_id,
                                  st.session_state.messages, title=new_title)
                st.success("Renamed!")

    with s2:
        st.markdown("#### ⚙️ Active Configuration")
        st.code(f"""LLM Model      : {GROQ_MODEL}
Embedding      : {EMBED_MODEL}
Reranker       : {RERANK_MODEL}
Chunk Size     : {CHUNK_SIZE} tokens
Rerank Top‑N   : {RERANK_TOP_N}
Multi‑Query N  : {MULTI_QUERY_N}
Groq Key       : {"✓ set" if os.getenv("GROQ_API_KEY")   else "✗ missing"}
Cohere Key     : {"✓ set" if os.getenv("COHERE_API_KEY") else "✗ missing"}""", language="text")

        st.markdown("#### 🏗️ Architecture")
        st.markdown("""
| Layer | Technology |
|---|---|
| LLM | Groq LPU · Llama 3.1 8B |
| Embeddings | HuggingFace · MiniLM-L6 |
| Vector DB | ChromaDB |
| Sparse Search | BM25Okapi |
| Retrieval | Hybrid Ensemble |
| Reranking | Cohere rerank-v3.5 |
| Multi-Query | LLM query expansion |
| Auth | SHA-256 + session |
| Audit | CSV append-only log |
| Conversations | Per-user JSON store |
| Formats | PDF, DOCX, CSV, TXT, MD |
""")
