"""RAG 评测 v2：要点覆盖率 + 引用页码正确率

用法： python eval_v2.py
输出： eval_report_v2.md
"""

import re
import time
from pathlib import Path

import rag as R

# (问题, 必须命中的要点, 正确答案所在页码 / None=知识库没有)
QUESTIONS = [
    ("年假天数怎么算？", ["5+n-1", "工龄"], 7),
    ("年假的有效期到什么时候？", ["2月28", "次年"], 7),
    ("病假期间工资怎么发？", ["80%", "最低工资"], 7),
    ("周末加班怎么补偿？", ["调休", "200"], 6),
    ("法定节假日加班给几倍工资？", ["300%"], 6),
    ("员工辞职需要提前多久申请？", ["30日"], 4),
    ("工资每月几号发？", ["10日", "次月"], 8),
    ("什么情况下不发年终奖？", ["12月1日", "离职"], 8),
    ("请假需要通过什么方式提交？", ["邮箱"], 7),
    ("对考勤记录有异议，几天内提出？", ["5日"], 6),
    ("迟到怎么扣钱？", None, None),  # 期望：拒答，且不引用
]

CONFIGS = [
    {"name": "A: 400/50/3", "chunk_size": 400, "overlap": 50, "top_k": 3},
    {"name": "B: 200/50/5", "chunk_size": 200, "overlap": 50, "top_k": 5},
]

CITE = re.compile(r"\[(\d+)\]")
PAGE = re.compile(r"第(\d+)(?:-(\d+))?页")


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def cited_pages(answer: str, hits: list[dict]) -> set[int]:
    pages = set()
    for m in CITE.finditer(answer):
        i = int(m.group(1)) - 1
        if 0 <= i < len(hits):
            for pm in PAGE.finditer(hits[i]["source"]):
                a = int(pm.group(1))
                b = int(pm.group(2) or a)
                pages.update(range(a, b + 1))
    return pages


def main():
    lines = ["# RAG 评测 v2（覆盖率 + 引用正确率）", ""]
    summary = []

    for cfg in CONFIGS:
        R.CONFIG.update(
            chunk_size=cfg["chunk_size"], overlap=cfg["overlap"], top_k=cfg["top_k"]
        )
        t0 = time.time()
        chunks = R.build_index(R.CONFIG["docs_dir"])
        has_pages = any("页" in c["source"] for c in chunks)
        lines.append(f"## {cfg['name']}")
        lines.append(f"- 块数：{len(chunks)}")

        cov_sum = cov_max = 0
        cite_ok = cite_total = 0
        rows = []

        for q, points, page in QUESTIONS:
            hits = R.search(q, chunks, cfg["top_k"])
            ans = R.ask(q, hits)
            na = norm(ans)

            if points is None:                      # 拒答题
                refused = ("未找到" in na) or ("没有找到" in na)
                has_cite = bool(CITE.search(ans))
                rows.append((q, "拒答", "PASS" if (refused and not has_cite) else "FAIL",
                             "无引用" if not has_cite else "仍带引用", ans))
                cite_total += 1
                cite_ok += 1 if (refused and not has_cite) else 0
                continue

            hit = [p for p in points if norm(p) in na]
            cov = len(hit) / len(points)
            cov_sum += cov
            cov_max += 1

            if has_pages:
                pages = cited_pages(ans, hits)
                ok = page in pages
                cite_total += 1
                cite_ok += 1 if ok else 0
                cite_label = f"引用页 {sorted(pages) or '无'} / 期望 {page}"
            else:
                ok = True
                cite_label = "N/A（语料无页码）"
            rows.append(
                (
                    q,
                    f"{len(hit)}/{len(points)}",
                    "PASS" if (cov == 1.0 and ok) else "FAIL",
                    cite_label,
                    ans,
                )
            )

        avg_cov = cov_sum / cov_max if cov_max else 0
        lines.append(f"- 平均要点覆盖率：{avg_cov:.0%}")
        lines.append(f"- 引用页码正确：{cite_ok}/{cite_total}（{cite_ok / cite_total:.0%}）")
        lines.append(f"- 耗时：{time.time() - t0:.1f}s")
        lines.append("")
        lines.append("| 问题 | 要点覆盖 | 判定 | 引用 |")
        lines.append("|---|---|---|---|")
        for q, c, v, p, _ in rows:
            lines.append(f"| {q} | {c} | {v} | {p} |")
        lines.append("")

        summary.append((cfg["name"], len(chunks), avg_cov, f"{cite_ok}/{cite_total}"))

    lines.append("## 汇总")
    lines.append("| 配置 | 块数 | 覆盖率 | 引用正确 |")
    lines.append("|---|---|---|---|")
    for name, n, cov, cite in summary:
        lines.append(f"| {name} | {n} | {cov:.0%} | {cite} |")

    Path("eval_report_v2.md").write_text("\n".join(lines), encoding="utf-8")
    print("done -> eval_report_v2.md")


if __name__ == "__main__":
    main()
