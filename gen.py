"""One two-column Japanese passage that carries a given time.

The hour is observed into the first sentence and the minute into the second,
under a language model steered by sequential Monte Carlo. Every other token is
sampled, and no other numeral may appear anywhere: on a clock face the only
numbers are the time. Column one is read first, so in 縦書き it stands on the
right.
"""
import asyncio, datetime, json, math, random, sys
from llamppl import CachedCausalLM, LMContext, Model, smc_standard

MODEL = "LiquidAI/LFM2.5-1.2B-JP-202606-MLX-4bit"
PARTICLES = 16
TEMP = 1.0
CAP = 15            # characters per column, so a sentence is at most 16 with its 。
DROP = 2            # glyphs the minute column hangs below the hour column
GAP = 1             # glyphs of clearance between the two numerals, measured across the columns
MAX_KANJI_RUN = 4   # Japanese prose rarely runs longer; Chinese always does
NUMERAL = set("0123456789〇一二三四五六七八九十百千零半")
END = ("。", "！", "？")
# 令和 began 1 May 2019; that year is 元年.
def era_date(d):
    y = d.year - 2018
    return f"令和{'元' if y == 1 else numeral(y)}年{numeral(d.month)}月{numeral(d.day)}日"

def minute_key(when): return when.strftime("%H:%M")

def is_kana(c): return "぀" <= c <= "ヿ" or c in "ーゝゞヽヾ"
def is_kanji(c): return "一" <= c <= "鿿" or c in "々〆ヶ"
def is_punct(c): return c in "、。「」『』（）！？…・―〜　"
def japanese(s): return bool(s) and all(is_kana(c) or is_kanji(c) or is_punct(c) for c in s)

KD = "零一二三四五六七八九"
def numeral(n):
    if n < 10: return KD[n]
    tens, ones = divmod(n, 10)
    return (KD[tens] if tens > 1 else "") + "十" + (KD[ones] if ones else "")

BRACKETS = set("「」『』（）")   # a 14-character column has no room to close one
TIME_COUNTERS = set("時分秒")   # a numeral followed by one of these reads as a clock

def masks(lm):
    """Tokens allowed anywhere; tokens allowed after a run of kanji; tokens that
    turn the numeral before them into a time."""
    jp, kanji_only, numerals, times = set(), set(), set(), set()
    for i, v in enumerate(lm.str_vocab):
        if any(c in NUMERAL or c in BRACKETS for c in v): numerals.add(i)
        if v and v[0] in TIME_COUNTERS: times.add(i)
        if japanese(v):
            jp.add(i)
            if all(is_kanji(c) for c in v): kanji_only.add(i)
    jp.add(lm.tokenizer.eos_token_id)
    return lm.token_mask(jp - numerals), lm.token_mask((jp - kanji_only) - numerals), lm.token_mask(times)

class TwoColumns(Model):
    """Writes both columns, or only the second when the first is given: the
    hour's sentence is then part of the prompt and `hour_span` names the
    numeral's slots in it."""
    def __init__(self, clock, prompt, targets, right=None, hour_span=None):
        super().__init__()
        self.clock = clock
        self.context = LMContext(clock.lm, prompt + (right or ""), temp=TEMP)
        self.targets = targets        # the hour and minute numerals
        self.ids = [clock.lm.tokenizer.encode(t, add_special_tokens=False) for t in targets]
        self.col = 0 if right is None else 1    # column being written
        self.pending = True           # this column's numeral not yet placed
        self.after_numeral = False    # the next token follows a numeral
        self.split = None if right is None else 0
        self.spans = [] if right is None else [tuple(hour_span)]

    def immutable_properties(self):
        return {"clock", "targets", "ids"}

    # Latents are drawn here, after the template is cloned per particle.
    async def start(self):
        self.due = [random.randint(2, 6), random.randint(2, 6)]
        self.col_tokens = 0

    # The two numerals must not sit level with each other across the columns:
    # the minute's slots, shifted down by the drop, stay clear of the hour's.
    def clear(self, col):
        if self.col == 0: return True
        h0, h1 = self.spans[0]
        m0 = len(col) + DROP
        m1 = m0 + len(self.targets[1])
        return m1 + GAP <= h0 or m0 >= h1 + GAP

    async def observe_ids(self, ids):
        for tid in ids:
            await self.observe(self.context.next_token(), tid)

    async def step(self):
        text = str(self.context)
        col = text if self.split is None else text[self.split:]
        # A numeral phrase opens after kana or punctuation: kanji-on-kanji forms a compound.
        boundary = bool(col) and (is_kana(col[-1]) or is_punct(col[-1]))

        if self.pending and self.col_tokens >= self.due[self.col] and boundary and self.clear(col):
            await self.observe_ids(self.ids[self.col])
            self.spans.append((len(text), len(str(self.context))))
            self.pending = False
            self.after_numeral = True
            return

        if not self.pending and len(col) >= CAP and not col.rstrip().endswith(END):
            await self.observe_ids([self.clock.end_id])       # end the sentence at the LM's price
            self.after_end()
            return

        kanji_run = 0
        for c in reversed(text):
            if not is_kanji(c): break
            kanji_run += 1
        mask = self.clock.break_mask if kanji_run >= MAX_KANJI_RUN else self.clock.mask
        if self.after_numeral: mask = mask & ~self.clock.time_mask
        await self.observe(self.context.mask_dist(mask), True)

        tok = await self.sample(self.context.next_token())
        self.col_tokens += 1
        self.after_numeral = False
        if tok.token_id == self.clock.eos:
            self.condition(False); self.finish(); return
        text = str(self.context)
        col = text if self.split is None else text[self.split:]
        if col.rstrip().endswith(END):
            if self.pending:
                if len(col) >= CAP: self.condition(False); self.finish()
                return
            self.after_end()

    def after_end(self):
        if self.col == 0:
            self.split = len(str(self.context))
            self.col_tokens = 0
            self.col, self.pending = 1, True
        else:
            self.finish()

class Clock:
    """The loaded model and everything derived from its vocabulary, built once."""
    def __init__(self):
        self.lm = CachedCausalLM.from_pretrained(MODEL, backend="mlx")
        self.mask, self.break_mask, self.time_mask = masks(self.lm)
        self.eos = self.lm.tokenizer.eos_token_id
        self.end_id = self.lm.tokenizer.encode("。", add_special_tokens=False)[0]

    async def generate(self, when, right=None, hour_span=None):
        """The passage for `when`, or None if no particle survived three draws.
        With `right`, the hour's sentence and its numeral's span from an earlier
        minute, only the minute's sentence is written."""
        hour, minute = numeral(when.hour % 12 or 12), numeral(when.minute)
        prompt = f"以下は、{era_date(when)}を描いた小説の一節である。\n\n"
        for attempt in range(3):
            program = TwoColumns(self, prompt, (hour, minute), right, hour_span)
            particles = await smc_standard(program, PARTICLES, ess_threshold=0.5)
            alive = [p for p in particles if math.isfinite(p.weight)]
            if not alive: continue
            best = max(alive, key=lambda p: p.weight)
            text = (right or "") + str(best.context)
            split = len(right) if right else best.split
            spans = [list(best.spans[0])] + [[a + split, b + split] for a, b in best.spans[1:]] if right \
                else [list(span) for span in best.spans]
            return {"time": minute_key(when), "right": text[:split], "left": text[split:],
                    "spans": spans, "split": split}
        return None

if __name__ == "__main__":
    now = datetime.datetime.now()
    if len(sys.argv) > 1:                                  # gen.py HH:MM, for looking at a time that is not now
        h, m = map(int, sys.argv[1].split(":"))
        now = now.replace(hour=h, minute=m)
    print(json.dumps(asyncio.run(Clock().generate(now)), ensure_ascii=False, indent=1))
