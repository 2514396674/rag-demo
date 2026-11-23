"""Streamlit 网页界面 —— 员工手册问答 Demo

启动：
    streamlit run app.py
或双击 run_web.cmd

特性：
- 每个浏览器标签页 = 独立会话（st.session_state），上下文互不干扰
- 支持连续追问（自动改写问题）
- 可展开查看检索到的原文片段
- 可调 top_k / 是否显示片段
"""

import os
import re

import streamlit as st

import rag as R

st.set_page_config(page_title="员工手册问答", page_icon="📘", layout="wide")


# ------------------------------------------------------------
# 索引只建一次，多个用户共享（只读，安全）
# ------------------------------------------------------------
@st.cache_resource(show_spinner="正在加载索引…")
def get_chunks():
    return R.build_index(R.CONFIG["docs_dir"])


@st.cache_resource(show_spinner=False)
def check_key() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())


chunks = get_chunks()
has_key = check_key()

# ------------------------------------------------------------
# 侧边栏
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📘 员工手册问答")
    st.caption("基于 RAG：先检索原文，再让模型作答，答案带页码引用。")

    st.metric("文档块数", len(chunks))

    top_k = st.slider("检索条数 top_k", 1, 8, R.CONFIG["top_k"])
    show_ctx = st.toggle("显示检索到的原文", value=False)
    st.divider()

    if not has_key:
        st.error("未检测到 DEEPSEEK_API_KEY\n\n请在环境变量里设置后重启。")
    else:
        st.success("模型已就绪")

    if st.button("清空当前对话"):
        st.session_state.messages = []
        st.rerun()

    st.caption("提示：这里每开一个浏览器标签页就是一个独立会话，互不串上下文。")

# ------------------------------------------------------------
# 会话状态（每个标签页独立）
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []          # 展示用
if "history" not in st.session_state:
    st.session_state.history = []           # 送模型用

st.title("员工手册问答")
st.caption("试试：「年假有几天？」→「那它和病假能叠加吗？」")

# 历史消息
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander(f"引用来源（{len(m['sources'])} 条）"):
                for i, s in enumerate(m["sources"], 1):
                    st.markdown(f"**[{i}] {s['source']}** · 相似度 {s['score']:.3f}")
                    st.code(s["text"][:400], language="text")

# 输入
if q := st.chat_input("输入你的问题…"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("检索并作答…"):
            rq = R.rewrite_question(q, st.session_state.history)
            hits = R.search(rq, chunks, top_k)
            ans = R.ask(q, hits, st.session_state.history)

        if rq != q:
            st.caption(f"（已改写为：{rq}）")

        st.markdown(ans)

        sources = []
        if has_key and not ans.startswith("[LLM 调用失败]"):
            sources = [{"source": h["source"], "score": h["score"], "text": h["text"]} for h in hits]

        if show_ctx:
            with st.expander(f"检索到的原文（top {len(hits)}）", expanded=True):
                for i, h in enumerate(hits, 1):
                    st.markdown(f"**[{i}] {h['source']}** · 相似度 {h['score']:.3f}")
                    st.code(h["text"][:400], language="text")
        elif sources:
            with st.expander(f"引用来源（{len(sources)} 条）"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**[{i}] {s['source']}** · 相似度 {s['score']:.3f}")
                    st.code(s["text"][:400], language="text")

    # 更新会话
    st.session_state.messages.append(
        {"role": "assistant", "content": ans, "sources": sources}
    )
    st.session_state.history.extend(
        [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
    )
    st.session_state.history = st.session_state.history[-R.CONFIG["history_rounds"] * 2 :]
