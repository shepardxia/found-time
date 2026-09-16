// The time, found inside a passage of Japanese prose generated for it once a
// minute (see gen.py). Two columns of 縦書き on the poster's right paper margin:
// the hour's sentence on the right, the minute's on the left, dropped two
// glyphs; only the numerals are black. Treatment 6c from the design pass.

export const command = "bash tick.sh";
export const refreshFrequency = 60000;

// The minute turns on a timer armed for the boundary itself: one wake per
// minute, at :00, re-armed from there. The data tick only carries new lines.
export const init = (dispatch) => {
  const arm = () => {
    const now = new Date();
    const wait = 60000 - (now.getSeconds() * 1000 + now.getMilliseconds());
    setTimeout(() => { dispatch({type: 'MINUTE', minute: minuteKey()}); arm(); }, wait);
  };
  arm();
};

const minuteKey = () => new Date().toTimeString().slice(0, 5);

export const initialState = {lines: {}, minute: minuteKey()};

export const updateState = (event, previous) => {
  if (event.type === 'MINUTE') return {...previous, minute: event.minute};
  try {
    return {...previous, lines: JSON.parse(event.output).lines};
  } catch (e) {
    return previous;
  }
};

// The line for the minute on the page's own clock, so the switch happens at
// the boundary rather than on the next tick; if that minute has not been
// written yet, the latest one that has.
const current = (lines, minute) => {
  if (lines[minute]) return lines[minute];
  const keys = Object.keys(lines).sort();
  return keys.length ? lines[keys[keys.length - 1]] : null;
};

const PITCH = 33;
const RIGHT_INSET = 6;      // from the screen's right edge, so it lands on the margin on any display
const HOUR_TOP = 130;
const MINUTE_TOP = 198;

export const className = `
  top: 0;
  right: ${RIGHT_INSET}px;
  width: ${2 * PITCH}px;
  font-family: "Hiragino Mincho ProN", "Noto Serif JP", serif;
  font-size: 28px;
  line-height: ${PITCH}px;
  letter-spacing: 0.22em;
  font-weight: 300;
  color: #b3b1ae;
`;

// Each column is pinned to the right edge on its own, so the hour's column
// never moves when the minute's changes. In vertical text line-height is the
// column width and letter-spacing the distance between glyphs; margin-inline
// is the pause before and after a glyph along the column.
const COLUMN = {position: 'absolute', writingMode: 'vertical-rl', textOrientation: 'mixed', whiteSpace: 'nowrap'};
const HOUR = {...COLUMN, right: 0, top: HOUR_TOP};
const MINUTE = {...COLUMN, right: PITCH, top: MINUTE_TOP};
const LIT = {
  fontWeight: 900,
  color: '#000',
  fontSize: 33,
  letterSpacing: '0.10em',
  marginInline: '0.40em',
};

// The numeral is one run, so it carries one pause before and one after.
const glyphs = (text, base, spans) => {
  const out = [];
  let at = 0;
  for (const [a, b] of spans) {
    if (b <= base || a >= base + text.length) continue;
    const from = a - base, to = b - base;
    if (from > at) out.push(<span key={out.length}>{text.slice(at, from)}</span>);
    out.push(<b key={out.length} style={LIT}>{text.slice(from, to)}</b>);
    at = to;
  }
  if (at < text.length) out.push(<span key={out.length}>{text.slice(at)}</span>);
  return out;
};

export const render = ({lines, minute}) => {
  const passage = current(lines, minute);
  if (!passage) return null;
  return (
    <div>
      <div style={HOUR}>{glyphs(passage.right, 0, passage.spans)}</div>
      <div style={MINUTE}>{glyphs(passage.left, passage.split, passage.spans)}</div>
    </div>
  );
};
