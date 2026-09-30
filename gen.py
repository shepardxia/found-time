"""One two-column Japanese passage that carries a given time.

The hour is found in the first sentence and the minute in the second. Each
sentence is the language model's own prose under sequential Monte Carlo: it
fits its column, carries its numeral once as a count, and shows no other
number. The prompt never names the time: it is the date and a few sentences
of a novel from the seed corpus, drawn afresh each time, which the passage
continues. Column one is read first, so in 縦書き it stands on the right.
"""
import asyncio, datetime, json, math, random, sys
from pathlib import Path
import numpy as np
from llamppl import CachedCausalLM, LMContext, Model, smc_standard
from llamppl.distributions import Bernoulli

MODEL = "LiquidAI/LFM2.5-1.2B-JP-202606-MLX-4bit"
PARTICLES = 32
TEMP = 1.0
CAP = 20                 # glyphs per column before its 。
WAIT = (2, 6)            # tokens of prose before a numeral, drawn per particle and column
LEAN = 5                 # with this much room left, the proposal leans toward ending the sentence
MAX_KANJI_RUN = 4        # Japanese prose rarely runs longer; Chinese always does
SEEDS = Path(__file__).with_name("corpus") / "corpus.txt"   # built by corpus/build.py
SEED_SENTENCES = 4       # consecutive sentences of one work that open the passage
NUMERAL = set("0123456789〇一二三四五六七八九十百千万億零半")
BRACKETS = set("「」『』（）")   # a column has no room to close one
TIME_COUNTERS = "時分秒"        # a numeral followed by one of these reads as a clock
ENDS = "。！？"

KD = "零一二三四五六七八九"
def numeral(n):
    if n < 10: return KD[n]
    tens, ones = divmod(n, 10)
    return (KD[tens] if tens > 1 else "") + "十" + (KD[ones] if ones else "")

# 令和 began 1 May 2019; that year is 元年.
def era_date(d):
    y = d.year - 2018
    return f"令和{'元' if y == 1 else numeral(y)}年{numeral(d.month)}月{numeral(d.day)}日"

def minute_key(when): return when.strftime("%H:%M")

def is_kana(c): return "぀" <= c <= "ヿ" or c in "ーゝゞヽヾ"
def is_kanji(c): return "一" <= c <= "鿿" or c in "々〆ヶ"
def is_punct(c): return c in "、。「」『』（）！？…・〜　"
def is_boundary(c): return is_kana(c) or is_punct(c)   # a numeral phrase opens after kana or punctuation
def japanese(s): return bool(s) and all(is_kana(c) or is_kanji(c) or is_punct(c) for c in s)

def kanji_run(text):
    n = 0
    for c in reversed(text):
        if not is_kanji(c): break
        n += 1
    return n

class Masks:
    """Token sets over the vocabulary, built once and shared by every particle."""
    def __init__(self, lm):
        vocab = lm.str_vocab
        ids = lambda pred: lm.token_mask(i for i, v in enumerate(vocab) if pred(v))
        prose = ids(lambda v: japanese(v) and not any(c in NUMERAL or c in BRACKETS for c in v))
        self.prose = prose
        self.broken = prose - ids(lambda v: all(is_kanji(c) for c in v))     # after a long kanji run
        self.time = ids(lambda v: v[:1] in TIME_COUNTERS)
        self.ends = ids(lambda v: any(c in ENDS for c in v))
        self.end_ids = list(self.ends & prose)
        self.fit = [ids(lambda v, r=r: len(v) <= r) for r in range(CAP + 1)]           # tokens within r glyphs
        self.period = lm.tokenizer.encode("。", add_special_tokens=False)[0]

class Passage(Model):
    """Two sentences, the hour's then the minute's, one per step. With `right`
    the hour's sentence is given, `hour_span` names its numeral's slots for
    the record, and only the minute's is written."""
    def __init__(self, clock, prompt, targets, right=None, hour_span=None):
        super().__init__()
        self.clock = clock
        self.context = LMContext(clock.lm, prompt + (right or ""), temp=TEMP)
        self.ids = [clock.lm.tokenizer.encode(t, add_special_tokens=False) for t in targets]
        self.glyphs = [len(t) for t in targets]
        self.columns = [right] if right else []
        self.spans = [tuple(hour_span)] if right else []   # each numeral's glyph slots within its column

    def immutable_properties(self):
        return {"clock", "ids", "glyphs"}

    async def observe_ids(self, ids):
        for tid in ids:
            await self.observe(self.context.next_token(), tid)

    async def step(self):
        col = len(self.columns)
        numeral, glyphs = self.ids[col], self.glyphs[col]
        m = self.clock.masks
        wait = random.randint(*WAIT)
        base = len(str(self.context))
        tokens, span, just_placed = 0, None, False
        while not self.finished:
            text = str(self.context)
            column = text[base:]
            n, room = len(column), CAP - len(column)
            if span is None and n and is_boundary(column[-1]) and n + glyphs <= CAP and tokens >= wait:
                await self.observe_ids(numeral)
                span, just_placed = (n, n + glyphs), True
                continue
            if column and column[-1] in ENDS: break
            if room == 0:
                if span is None: self.condition(False); return
                await self.observe_ids([m.period])
                break
            mask = m.broken if kanji_run(text) >= MAX_KANJI_RUN else m.prose
            if just_placed: mask = mask - m.time
            if span is None:
                mask = (mask - m.ends) & m.fit[room - glyphs]
            else:
                mask = mask & m.fit[room]
                if room <= LEAN:
                    # Ask the model whether the sentence ends here, proposing yes at least 1/room
                    # of the time; the importance weight keeps the model's own answer the target.
                    p_end = np.exp(np.logaddexp.reduce(self.context.next_token_logprobs[m.end_ids]))
                    await self.sample(self.context.mask_dist(m.ends & mask),
                                      proposal=Bernoulli(max(min(p_end, 1.0), 1 / room)))
            await self.observe(self.context.mask_dist(mask), True)
            await self.sample(self.context.next_token())
            tokens, just_placed = tokens + 1, False
        if self.finished: return
        self.columns.append(str(self.context)[base:])
        self.spans.append(span)
        if len(self.columns) == 2: self.finish()

class Clock:
    """The loaded model and everything derived from its vocabulary, built once."""
    def __init__(self):
        self.lm = CachedCausalLM.from_pretrained(MODEL, backend="mlx")
        self.masks = Masks(self.lm)
        self.works = [w.splitlines() for w in SEEDS.read_text(encoding="utf-8").split("\n\n")]

    def prompt(self, when):
        work = random.choice(self.works)
        i = random.randrange(max(1, len(work) - SEED_SENTENCES + 1))
        return f"以下は、{era_date(when)}を描いた小説の一節である。\n\n" + "".join(work[i:i + SEED_SENTENCES])

    async def generate(self, when, right=None, hour_span=None, tries=3):
        """The passage for `when`, or None if no particle survived `tries` runs.
        With `right`, the hour's sentence and its numeral's span from earlier
        in the hour, only the minute's sentence is written."""
        hour, minute = numeral(when.hour % 12 or 12), numeral(when.minute)
        for _ in range(tries):
            program = Passage(self, self.prompt(when), (hour, minute), right, hour_span)
            particles = await smc_standard(program, PARTICLES, ess_threshold=0.5)
            self.lm.clear_cache()                      # the backend keeps every prefix it has seen
            alive = [p for p in particles if math.isfinite(p.weight)]
            if not alive: continue
            top = max(p.weight for p in alive)
            p = random.choices(alive, weights=[math.exp(p.weight - top) for p in alive])[0]
            split = len(p.columns[0])
            (a, b), (c, d) = p.spans
            return {"time": minute_key(when), "right": p.columns[0], "left": p.columns[1],
                    "spans": [[a, b], [c + split, d + split]], "split": split}
        return None

if __name__ == "__main__":
    now = datetime.datetime.now()
    if len(sys.argv) > 1:                                  # gen.py HH:MM, for looking at a time that is not now
        h, m = map(int, sys.argv[1].split(":"))
        now = now.replace(hour=h, minute=m)
    print(json.dumps(asyncio.run(Clock().generate(now)), ensure_ascii=False, indent=1))
