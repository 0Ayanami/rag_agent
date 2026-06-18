"""
批量抓取东方财富某机构发布页的报告并下载 PDF。
Features:
- 解析页面中 body -> main -> main-content -> framecontent -> org_publishtable 的 table（通过 id/name/class 尝试定位）
- 按序号、报告类型、研究对象、日期区间筛选
- 并发下载（ThreadPoolExecutor）
- 多种方式查找 PDF 链接（a[href$=.pdf], iframe/embed/object 指向 pdf, 自定义 a.pdf-link 等）
- 支持 dry-run 模式打印将下载的项
"""

import os
import re
import json
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlsplit
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional, Dict, Any

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
DEFAULT_TIMEOUT = 20

def safe_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:200] if len(name) > 200 else name

def extract_pdf_url(page_url: str, html: str) -> Optional[str]:
    """
    从报告页面 HTML 中找到 PDF 链接，返回绝对 URL 或 None
    """
    soup = BeautifulSoup(html, "lxml")
    link = soup.select_one("a.pdf-link[href]")
    if link:
        href = (link.get("href") or "").strip()
        if href:
            return urljoin(page_url, href)
    return None


def download_pdf_from_report(report_page_url: str, outdir: str, title: str, session: requests.Session, verbose: bool = False) -> Optional[str]:
    """
    访问报告详情页，寻找 PDF 并下载。返回保存路径或 None。
    """
    try:
        resp = session.get(report_page_url, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        if verbose:
            print(f"[ERROR] 请求报告页失败: {report_page_url} -> {e}")
        return None

    pdf_url = extract_pdf_url(report_page_url, resp.text)
    if not pdf_url:
        if verbose:
            print(f"[WARN] 未在报告页找到 PDF 链接: {report_page_url}")
        return None

    pdf_headers = dict(HEADERS)
    pdf_headers["Referer"] = report_page_url
    try:
        pdf_resp = session.get(pdf_url, headers=pdf_headers, timeout=DEFAULT_TIMEOUT * 2)
        pdf_resp.raise_for_status()
    except Exception as e:
        if verbose:
            print(f"[ERROR] 下载 PDF 失败: {pdf_url} -> {e}")
        return None

    if (not pdf_resp.content) or (len(pdf_resp.content) < 5) or (not pdf_resp.content.startswith(b"%PDF")):
        if verbose:
            ct = pdf_resp.headers.get("Content-Type")
            print(f"[WARN] 非有效 PDF 内容: url={pdf_url} content_type={ct} bytes={len(pdf_resp.content)}")
        return None

    try:
        s_title = safe_filename(title)
    except Exception:
        s_title = safe_filename(os.path.basename(urlparse(pdf_url).path) or "report")

    filename = f"{s_title}.pdf"
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)

    i = 1
    base, ext = os.path.splitext(path)
    while os.path.exists(path):
        path = f"{base}_{i}{ext}"
        i += 1

    with open(path, "wb") as f:
        f.write(pdf_resp.content)

    if verbose:
        print(f"[OK] 下载完成: {path}")
    return path

def parse_rows_from_initdata(html: str, base_url: str) -> List[Dict[str, str]]:
    """
    页面无静态 table 时，从 `var initdata = {...};` 解析首屏数据。
    """
    m = re.search(r"var\s+initdata\s*=\s*(\{.*?\});", html, flags=re.S)
    if not m:
        return []

    try:
        payload = json.loads(m.group(1))
    except Exception:
        return []

    data = payload.get("data") or []
    rows: List[Dict[str, str]] = []

    for i, item in enumerate(data, 1):
        info_code = str(item.get("infoCode") or "").strip()
        report_url = urljoin(base_url, f"/report/info/{info_code}.html") if info_code else ""

        author = item.get("researcher") or ""
        if not author:
            a = item.get("author")
            if isinstance(a, list):
                author = ",".join([str(x) for x in a if x is not None])
            elif a is not None:
                author = str(a)

        publish_date = str(item.get("publishDate") or "")
        if publish_date:
            # 提取出具体的日期部分，去掉时间等多余信息
            publish_date = publish_date.split(" ")[0]

        rows.append(
            {
                "index": str(i),
                "title": str(item.get("title") or ""),
                "type": str(item.get("columnType") or "").strip(),
                "target": str(item.get("stockName") or item.get("industryName") or ""),
                "author": str(author),
                "org": str(item.get("orgSName") or ""),
                "date": publish_date,
                "report_url": report_url,
            }
        )
    return rows

def parse_date(datestr: str) -> Optional[datetime]:
    if not datestr:
        return None
    datestr = datestr.strip()
    # 常见格式尝试
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y年%m月%d日",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(datestr, fmt)
        except Exception:
            continue
    # 尝试只抽取 YYYY-MM-DD
    m = re.search(r"(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})", datestr)
    if m:
        s = m.group(1).replace(".", "-").replace("/", "-")
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            pass
    # 最后尝试仅取年份
    m2 = re.search(r"(\d{4})", datestr)
    if m2:
        try:
            return datetime(int(m2.group(1)), 1, 1)
        except Exception:
            pass
    return None


def match_filters(row: Dict[str, str], filters: Dict[str, Any]) -> bool:
    # 筛选出需要下载的报告
    if filters.get("indexes"):
        try:
            digits = re.search(r"(\d+)", row.get("index", ""))
            if not digits:
                return False
            idx = int(digits.group(1))
            if idx not in filters["indexes"]:
                return False
        except Exception:
            return False

    if filters.get("types"):
        v = row.get("type", "")
        ok = False
        for t in filters["types"]:
            if t in v:
                ok = True
                break
        if not ok:
            return False

    if filters.get("targets"):
        v = row.get("target", "")
        ok = False
        for t in filters["targets"]:
            if t in v:
                ok = True
                break
        if not ok:
            return False

    if filters.get("date_from") or filters.get("date_to"):
        d = parse_date(row.get("date", "")) or None
        if d is None:
            return False
        if filters.get("date_from") and d < filters["date_from"]:
            return False
        if filters.get("date_to") and d > filters["date_to"]:
            return False

    return True


def parse_indexes_arg(arg: Optional[str]) -> Optional[set]:
    # 解析 indexes 参数，支持范围如 1-10,20,30-40
    if not arg:
        return None
    s = set()
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            try:
                a_i = int(a)
                b_i = int(b)
                for i in range(min(a_i, b_i), max(a_i, b_i) + 1):
                    s.add(i)
            except Exception:
                continue
        else:
            try:
                s.add(int(p))
            except Exception:
                continue
    return s if s else None


def run_batch(org_url: str, outdir: str, indexes_arg: Optional[str], types_arg: Optional[str],
              targets_arg: Optional[str], date_from_arg: Optional[str], date_to_arg: Optional[str],
              concurrency: int = 4, dry_run: bool = False, verbose: bool = False):
    session = requests.Session()
    try:
        resp = session.get(org_url, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"请求机构页面失败: {org_url} -> {e}")

    # 读取表格
    parsed_url = urlsplit(org_url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

    rows = parse_rows_from_initdata(resp.text, base_url)
    if verbose:
        print(f"[INFO] 通过 initdata 解析到 {len(rows)} 行")
    
    if not rows:
        raise RuntimeError("未找到可解析的数据（table 与 initdata 均为空），请检查页面结构或网络返回内容。")
    if verbose:
        print(f"[INFO] 总解析行数: {len(rows)}")

    # 解析筛选参数
    filters = {}
    idx_set = parse_indexes_arg(indexes_arg)
    if idx_set:
        filters["indexes"] = idx_set
    if types_arg:
        filters["types"] = [t.strip() for t in types_arg.split(",") if t.strip()]
    if targets_arg:
        filters["targets"] = [t.strip() for t in targets_arg.split(",") if t.strip()]
    if date_from_arg:
        filters["date_from"] = parse_date(date_from_arg)
    if date_to_arg:
        filters["date_to"] = parse_date(date_to_arg)

    tasks = []
    for r in rows:
        if match_filters(r, filters):
            tasks.append(r)

    if not tasks:
        print("[INFO] 没有找到匹配条件的报告条目。")
        return

    print(f"[INFO] 将下载 {len(tasks)} 个报告（concurrency={concurrency}，dry_run={dry_run}）")
    if dry_run:
        for t in tasks:
            print(f" - index={t.get('index')} date={t.get('date')} type={t.get('type')} target={t.get('target')} url={t.get('report_url')}")
        return

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        future_to_row = {}
        for r in tasks:
            future = ex.submit(download_pdf_from_report, r["report_url"], outdir, r["title"], session, verbose)
            future_to_row[future] = r

        for fut in as_completed(future_to_row):
            r = future_to_row[fut]
            try:
                path = fut.result()
                results.append((r, path))
            except Exception as e:
                print(f"[ERROR] 下载任务异常: {r.get('report_url')} -> {e}")
                results.append((r, None))

    ok = sum(1 for _, p in results if p)
    fail = len(results) - ok
    print(f"[DONE] 成功 {ok}，失败 {fail}，保存目录: {os.path.abspath(outdir)}")
    if verbose and fail > 0:
        for r, p in results:
            if not p:
                print(f"  [FAIL] index={r.get('index')} url={r.get('report_url')}")


def main_cli():
    parser = argparse.ArgumentParser(description="从东方财富机构发布页批量下载 PDF（支持筛选）")
    parser.add_argument("org_url", help="机构发布列表页 URL，例如: https://data.eastmoney.com/report/orgpublish.jshtml?orgcode=80000031")
    parser.add_argument("--outdir", "-o", default="data/eastmoney_reports", help="保存目录")
    parser.add_argument("--indexes", default="1-20", help="按序号筛选，例如: 1,2,5-10")
    parser.add_argument("--types", help="按报告类型筛选，逗号分隔")
    parser.add_argument("--targets", help="按研究对象/公司筛选，逗号分隔")
    parser.add_argument("--date-from", help="开始日期（含），格式例子: 2026-01-01")
    parser.add_argument("--date-to", help="结束日期（含")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="并发下载数")
    parser.add_argument("--dry-run", action="store_true", help="只列出将下载的报告，不实际下载")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()

    run_batch(
        org_url=args.org_url,
        outdir=args.outdir,
        indexes_arg=args.indexes,
        types_arg=args.types,
        targets_arg=args.targets,
        date_from_arg=args.date_from,
        date_to_arg=args.date_to,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main_cli()
