"""Builds corpus.txt: clean sentences from Aozora Bunko, one per line, in order.

Six authors, modern orthography, public domain. Ruby and editor's notes are
stripped; a sentence is kept only if it could stand on the clock -- kana,
kanji, 、 and 。 and nothing else -- and the first stretch of every work is
skipped, so a seed is never a famous opening.
"""
import csv, io, json, random, re, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen, Request

HERE = Path(__file__).resolve().parent
INDEX = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
AUTHORS = {("梶井", "基次郎"), ("宮沢", "賢治"), ("夏目", "漱石"), ("芥川", "竜之介"), ("太宰", "治"), ("中島", "敦")}
PER_AUTHOR = 50
SKIP_HEAD = 0.05
MIN, MAX = 14, 60
NUMERAL = set("〇一二三四五六七八九十百千万億零半")

def is_kana(c): return "぀" <= c <= "ヿ" or c in "ーゝゞヽヾ"
def is_kanji(c): return "一" <= c <= "鿿" or c in "々〆ヶ"
def clean(s): return all((is_kana(c) or is_kanji(c) or c in "、。") and c not in NUMERAL for c in s)

def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "found-time/1 (personal desktop clock)"}), timeout=60) as r:
        return r.read()

def index():
    z = zipfile.ZipFile(io.BytesIO(fetch(INDEX)))
    rows = csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8-sig"))
    picked = {}
    for r in rows:
        if (r["姓"], r["名"]) not in AUTHORS or r["役割フラグ"] != "著者": continue
        if r["文字遣い種別"] != "新字新仮名" or r["作品著作権フラグ"] != "なし": continue
        url = r["テキストファイルURL"]
        if not url.endswith(".zip"): continue
        picked.setdefault(r["姓"] + r["名"], {})[r["作品ID"]] = (r["作品名"], url)
    return picked

def body(raw):
    text = raw.decode("cp932", errors="ignore")
    parts = re.split(r"\n-{10,}\n", text)
    text = parts[-1] if len(parts) >= 3 else text            # the header sits between two rules
    text = text.split("\n底本：")[0]
    text = re.sub(r"《[^》]*》", "", text)                     # ruby
    text = re.sub(r"［＃[^］]*］", "", text)                   # editor's notes
    text = text.replace("｜", "")
    return re.sub(r"\s+", "", text)

def sentences(text):
    out = []
    for s in re.split(r"(?<=[。！？])", text):
        s = s.strip()
        if MIN <= len(s) <= MAX and clean(s) and s[-1] == "。":
            out.append(s)
    return out

def work(author, title, url):
    try:
        z = zipfile.ZipFile(io.BytesIO(fetch(url)))
        name = next(n for n in z.namelist() if n.endswith(".txt"))
        sents = sentences(body(z.read(name)))
        sents = sents[int(len(sents) * SKIP_HEAD):]
        time.sleep(0.2)
        return {"author": author, "title": title, "sentences": sents} if len(sents) >= 10 else None
    except Exception as e:
        print(f"  skip {title}: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    random.seed(0)
    picked = index()
    jobs = []
    for author, works in picked.items():
        items = list(works.values())
        random.shuffle(items)
        jobs += [(author, t, u) for t, u in items[:PER_AUTHOR]]
        print(f"{author}: {len(works)} works, taking {min(PER_AUTHOR, len(works))}")
    with ThreadPoolExecutor(4) as pool:
        records = [r for r in pool.map(lambda j: work(*j), jobs) if r]
    lines = []
    for r in records:
        lines += r["sentences"] + [""]                       # a blank line between works
    (HERE / "corpus.txt").write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    n = sum(len(r["sentences"]) for r in records)
    print(f"{len(records)} works, {n} sentences -> corpus.txt ({(HERE / 'corpus.txt').stat().st_size // 1024} KB)")
