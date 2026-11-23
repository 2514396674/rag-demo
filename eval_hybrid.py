"""对比：纯向量检索 vs 混合检索（向量 + BM25）

用法： python eval_hybrid.py
输出： eval_hybrid.md
"""

import re
import time
from pathlib import Path

import rag as R

# (问题, 要点, 正确页码)
QUESTIONS = [
    ("年假天数怎么算？", ["5+n-1"], 7),
    ("月薪折算日工资的系数是多少？", ["21.75"], 7),
    ("试用期考核不合格，提前几天书面通知？", ["3"], 3),
    ("年度调休结余最多保留几天？", ["6"], 6),
    ("单次调休最长多少天？", ["15"], 6),
    ("对考勤记录有异议，几天内提出？", ["5"], 6),
    ("工资每月几号发？", ["10"], 8),
    ("离职未结算工资什么时候发？", ["15"], 5),
]

MODES = [
    {"name": "纯向量", "hybrid": False},
    {"name": "混合检索", "hybrid": True},
]

CITE = re.compile(r"\[(\d+)\]")
PAGE = re.compile(r"第(\d+)(?:-(\d+))?页")


def norm(s):
    return re.sub(r"\s+", "", s or "")


def cited_pages(ans, hits):
    pages = set()
    for m in CITE.finditer(ans):
        i = int(m.group(1)) - 1
        if 0 <= i < len(hits):
            for pm in PAGE.finditer(hits[i]["source"]):
                a = int(pm.group(1))
                b = int(pm.group(2) or a)
                pages.update(range(a, b + 1))
    return pages


def main():
    R.CONFIG.update(chunk_size=400, overlap=50, top_k=3)
    chunks = R.build_index(R.CONFIG["docs_dir"])
    has_pages = any("页" in c["source"] for c in chunks)
    lines = ["# 纯向量 vs 混合检索", "", f"块数：{len(chunks)}", ""]
    summary = []

    for mode in MODES:
        R.CONFIG["hybrid"] = mode["hybrid"]
        t0 = time.time()
        cov_ok = cite_ok = 0
        rows = []
        for q, points, page in QUESTIONS:
            hits = R.search(q, chunks, R.CONFIG["top_k"])
            ans = R.ask(q, hits)
            na = norm(ans)
            cov = all(norm(p) in na for p in points)
            pages = cited_pages(ans, hits)
            cite = (page in pages) if has_pages else True
            cov_ok += 1 if cov else 0
            cite_ok += 1 if cite else 0
            rows.append((q, "✅" if cov else "❌", "✅" if cite else "❌", ans.replace("\n", " ")[:60]))
        n = len(QUESTIONS)
        lines.append(f"## {mode['name']}")
        lines.append(f"- 要点正确：{cov_ok}/{n}（{cov_ok/n:.0%}）")
        lines.append(f"- 引用正确：{cite_ok}/{n}（{cite_ok/n:.0%}）")
        lines.append(f"- 耗时：{time.time()-t0:.1f}s")
        lines.append("")
        lines.append("| 问题 | 要点 | 引用 | 回答摘要 |")
        lines.append("|---|---|---|---|")
        for r in rows:
            lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
        lines.append("")
        summary.append((mode["name"], cov_ok, cite_ok, n))

    lines.append("## 汇总")
    lines.append("| 模式 | 要点正确 | 引用正确 |")
    lines.append("|---|---|---|")
    for name, c, ci, n in summary:
        lines.append(f"| {name} | {c}/{n} | {ci}/{n} |")

    Path("eval_hybrid.md").write_text("\n".join(lines), encoding="utf-8")
    print("done -> eval_hybrid.md")


if __name__ == "__main__":
    main()
