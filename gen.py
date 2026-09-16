"""Prints one two-column Japanese passage that carries the current time.

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

def masks(lm):
    jp, kanji_only, numerals = set(), set(), set()
    for i, v in enumerate(lm.str_vocab):
        if any(c in NUMERAL or c in BRACKETS for c in v): numerals.add(i)
        if japanese(v):
            jp.add(i)
            if all(is_kanji(c) for c in v): kanji_only.add(i)
    jp.add(lm.tokenizer.eos_token_id)
    return jp - numerals, (jp - kanji_only) - numerals

class TwoColumns(Model):
    def __init__(self, lm, prompt, hour, minute, mask, break_mask, end_id):
        super().__init__()
        self.lm, self.mask, self.break_mask, self.end_id = lm, mask, break_mask, end_id
        self.context = LMContext(lm, prompt, temp=TEMP)
        self.eos = lm.tokenizer.eos_token_id
        self.targets = (hour, minute)
        self.phase = 0            # 0 hour pending, 1 ending column one, 2 minute pending, 3 ending column two
        self.split = None
        self.spans = []

    def immutable_properties(self):
        return {"lm", "eos", "mask", "break_mask", "end_id"}

    # Latents are drawn here, after the template is cloned per particle.
    async def start(self):
        self.due = [random.randint(2, 6), random.randint(2, 6)]
        self.ids = [self.lm.tokenizer.encode(t, add_special_tokens=False) for t in self.targets]
        self.col_tokens = 0

    # The two numerals must not sit level with each other across the columns:
    # the minute's slots, shifted down by the drop, stay clear of the hour's.
    def clear(self, col):
        if self.phase != 2: return True
        h0, h1 = self.spans[0]
        hour = range(h0, h1)                                   # glyph slots in column one
        m0 = len(col) + DROP
        minute = range(m0, m0 + len(self.targets[1]))
        return minute.stop + GAP <= hour.start or minute.start >= hour.stop + GAP

    def column(self):
        text = str(self.context)
        return text if self.split is None else text[self.split:]

    def kanji_run(self):
        n = 0
        for c in reversed(str(self.context)):
            if is_kanji(c): n += 1
            else: break
        return n

    async def observe_ids(self, ids):
        for tid in ids:
            await self.observe(self.context.next_token(), tid)

    async def step(self):
        col = self.column()
        pending = self.phase in (0, 2)
        # A numeral phrase opens after kana or punctuation: kanji-on-kanji forms a compound.
        boundary = bool(col) and (is_kana(col[-1]) or is_punct(col[-1]))

        if pending and self.col_tokens >= self.due[self.phase // 2] and boundary and self.clear(col):
            before = len(str(self.context))
            await self.observe_ids(self.ids[self.phase // 2])
            self.spans.append((before, len(str(self.context))))
            self.phase += 1
            return

        if not pending and len(col) >= CAP and not col.rstrip().endswith(END):
            await self.observe_ids([self.end_id])       # end the sentence at the LM's price
            self.after_end()
            return

        mask = self.break_mask if self.kanji_run() >= MAX_KANJI_RUN else self.mask
        await self.observe(self.context.mask_dist(mask), True)

        tok = await self.sample(self.context.next_token())
        self.col_tokens += 1
        if tok.token_id == self.eos:
            self.condition(False); self.finish(); return
        if self.column().rstrip().endswith(END):
            if pending:
                if len(self.column()) >= CAP: self.condition(False); self.finish()
                return
            self.after_end()

    def after_end(self):
        if self.phase == 1:
            self.split = len(str(self.context))
            self.col_tokens = 0
            self.phase = 2
        else:
            self.finish()

class Clock:
    """The loaded model and everything derived from its vocabulary, built once."""
    def __init__(self):
        self.lm = CachedCausalLM.from_pretrained(MODEL, backend="mlx")
        self.mask, self.break_mask = masks(self.lm)
        self.end_id = self.lm.tokenizer.encode("。", add_special_tokens=False)[0]

    async def generate(self, when):
        hour, minute = numeral(when.hour % 12 or 12), numeral(when.minute)
        # The frame names the two numerals so that the observed ones arrive where the
        # model expected a number -- as counts, not as a clock.
        prompt = (f"以下は、{era_date(when)}を描いた小説の一節である。"
                  f"文中には{hour}と{minute}、二つの数が現れる。\n\n")
        for attempt in range(3):                          # another draw if none survived
            particles = await smc_standard(
                TwoColumns(self.lm, prompt, hour, minute, self.mask, self.break_mask, self.end_id),
                PARTICLES, ess_threshold=0.5)
            alive = [p for p in particles if math.isfinite(p.weight)]
            if not alive: continue
            best = max(alive, key=lambda p: p.weight)
            text = str(best.context)
            return {"time": when.strftime("%H:%M"), "hour": hour, "minute": minute,
                    "right": text[:best.split], "left": text[best.split:],
                    "spans": [[a, b] for a, b in best.spans], "split": best.split}
        return {"time": when.strftime("%H:%M"), "hour": hour, "minute": minute, "error": "no particle survived"}

if __name__ == "__main__":
    now = datetime.datetime.now()
    if len(sys.argv) > 1:                                  # gen.py HH:MM, for looking at a time that is not now
        h, m = map(int, sys.argv[1].split(":"))
        now = now.replace(hour=h, minute=m)
    print(json.dumps(asyncio.run(Clock().generate(now)), ensure_ascii=False, indent=1))
