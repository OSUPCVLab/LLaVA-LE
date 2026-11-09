"""Render LLaVA-LE demo GIFs (caption scene + typed Q&A rounds).

Reproduces the layout, colours and timing of docs/assets/llava_le_demo*.gif:
1560x1400 px, blue header, an image -> "Caption:" scene that types word by word,
then one scene per Q&A round (question typed, short pause, answer typed).

Usage:
    python docs/tools/make_demo_gif.py docs/tools/demo_examples.json
    python docs/tools/make_demo_gif.py docs/tools/demo_examples.json --preview   # also dumps key frames as PNG

Config (paths are relative to the config file):
{
  "examples": [
    {
      "json": "examples/N36.600_E179.000_r02_c11.json",   # from fetch_lucid_example.py
      "image": "examples/N36.600_E179.000_r02_c11.png",   # optional, defaults to json stem + .png
      "caption_sentences": 2,          # optional: keep only the first N caption sentences
      "caption": "...",                # optional: override caption text
      "qa_rounds": [0, 1],             # optional: which QA pairs (0-based) to show, default [0, 1];
                                       #   [] = caption-only GIF
      "show_caption": false,           # optional: skip the caption scene (Q&A-only GIF), default true
      "qa": [["q", "a"], ...],         # optional: override QA pairs entirely
      "out": "../assets/demo_new/llava_le_demo_custom_1.gif"
    }
  ]
}
"""

import argparse
import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

# ── Canvas & palette (sampled from the original GIFs) ──
W, H = 1560, 1400
BG = (248, 249, 250)
HEADER_BG = (13, 71, 161)
HEADER_H = 106
TEXT = (33, 33, 33)
GREEN = (46, 125, 50)
BLUE = (21, 101, 192)
GREY = (117, 117, 117)
BOT_BUBBLE = (232, 245, 233)
USER_BUBBLE = (227, 242, 253)
IMG_BORDER = (170, 170, 170)
ARROW = (140, 140, 140)

# ── Layout ──
MARGIN_X = 60
BUBBLE_RIGHT = 1400
BUBBLE_PAD_X, BUBBLE_PAD_Y = 40, 18
BUBBLE_RADIUS = 18
EMOJI_SIZE = 92
LABEL_GAP = 12           # emoji bottom -> bubble top
SECTION_GAP = 60         # user bubble bottom -> bot emoji top
BOTTOM_LIMIT = H - 50

# ── Timing (ms) — identical to the originals ──
T_CAPTION_WORD = 300
T_CAPTION_HOLD = 3000
T_QA_WORD = 200
T_QUESTION_HOLD = 1600
T_ROUND_HOLD = 3000
T_FINAL_HOLD = 6000

USER_EMOJI = "\U0001F466"   # 👦
BOT_EMOJI = "\U0001F916"    # 🤖

# ── Fonts (Windows Segoe UI, falling back to DejaVu / Noto on Linux) ──
FONT_CANDIDATES = {
    "regular": ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "bold": ["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "emoji": ["C:/Windows/Fonts/seguiemj.ttf", "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
              "/System/Library/Fonts/Apple Color Emoji.ttc"],
}


def find_font(kind):
    for p in FONT_CANDIDATES[kind]:
        if os.path.exists(p):
            return p
    raise SystemExit(f"No {kind} font found; edit FONT_CANDIDATES in {__file__}")


FONT_REG, FONT_BOLD, FONT_EMOJI = find_font("regular"), find_font("bold"), find_font("emoji")
_font_cache, _emoji_cache = {}, {}


def font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def emoji(ch, size):
    """Render a colour emoji at its native size, then scale (works for COLR and bitmap fonts)."""
    key = (ch, size)
    if key not in _emoji_cache:
        native = 109  # NotoColorEmoji only renders at 109; fine for Segoe/Apple too
        f = ImageFont.truetype(FONT_EMOJI, native)
        tmp = Image.new("RGBA", (native * 2, native * 2), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((native // 2, native // 2), ch, font=f, embedded_color=True)
        bbox = tmp.getbbox()
        tmp = tmp.crop(bbox) if bbox else tmp
        _emoji_cache[key] = tmp.resize((size, size), Image.LANCZOS)
    return _emoji_cache[key]


# ── Text helpers ──
def wrap(text, fnt, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}" if cur else word
        if fnt.getlength(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def first_sentences(text, n):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n])


class Style:
    """Font sizes; scaled down uniformly if an example would overflow the canvas."""

    def __init__(self, scale=1.0):
        self.body = font(FONT_REG, round(56 * scale))
        self.label = font(FONT_BOLD, round(60 * scale))
        self.round = font(FONT_BOLD, round(50 * scale))
        self.line_h = round(84 * scale)
        self.cursor_w = max(4, round(26 * scale))


def bubble_size(lines, st):
    tw = max((st.body.getlength(l) for l in lines), default=0) + st.cursor_w + 10
    return tw + 2 * BUBBLE_PAD_X, len(lines) * st.line_h + 2 * BUBBLE_PAD_Y


def draw_bubble(d, x, y, lines, fill, st, cursor):
    bw, bh = bubble_size(lines or [""], st)
    d.rounded_rectangle([x, y, x + bw, y + bh], BUBBLE_RADIUS, fill=fill)
    ty = y + BUBBLE_PAD_Y
    for i, line in enumerate(lines):
        d.text((x + BUBBLE_PAD_X, ty + i * st.line_h), line, font=st.body, fill=TEXT)
    if cursor:
        last = lines[-1] if lines else ""
        cx = x + BUBBLE_PAD_X + st.body.getlength(last) + (8 if last else 0)
        cy = ty + (len(lines) - 1 if lines else 0) * st.line_h
        asc, desc = st.body.getmetrics()
        d.rectangle([cx, cy + asc * 0.25, cx + st.cursor_w, cy + asc + desc * 0.4], fill=TEXT)
    return y + bh


def draw_label(img, d, x, y, ch, text, color, st):
    img.paste(e := emoji(ch, EMOJI_SIZE), (x, y), e)
    _, desc = st.label.getmetrics()
    d.text((x + EMOJI_SIZE + 14, y + EMOJI_SIZE - desc + 4), text, font=st.label, fill=color, anchor="ls")
    return y + EMOJI_SIZE + LABEL_GAP


def base_frame():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, HEADER_H], fill=HEADER_BG)
    # header robot icon (white face, antenna, blue eyes/mouth)
    cx, cy, r = 64, 60, 26
    d.ellipse([cx - r, cy - r + 6, cx + r, cy + r + 6], fill="white")
    d.line([cx, cy - r + 6, cx, cy - r - 8], fill="white", width=5)
    d.ellipse([cx - 7, cy - r - 16, cx + 7, cy - r - 2], fill="white")
    for ex in (cx - 10, cx + 10):
        d.ellipse([ex - 5, cy + 2, ex + 5, cy + 12], fill=HEADER_BG)
    d.rectangle([cx - 10, cy + 19, cx + 10, cy + 23], fill=HEADER_BG)
    d.text((116, HEADER_H // 2), "LLaVA-LE", font=font(FONT_BOLD, 64), fill="white", anchor="lm")
    return img, d


# ── Scenes ──
def caption_layout(patch, caption, st):
    bx = 518
    max_w = BUBBLE_RIGHT - bx - 2 * BUBBLE_PAD_X - st.cursor_w - 10
    lines = wrap(caption, st.body, max_w)
    bottom = 138 + EMOJI_SIZE + LABEL_GAP + bubble_size(lines, st)[1]
    return bx, max_w, bottom


def caption_frame(patch, words, st, cursor):
    img, d = base_frame()
    ix, iy, isz = 58, 148, 334
    d.rectangle([ix - 2, iy - 2, ix + isz + 1, iy + isz + 1], fill=IMG_BORDER)
    img.paste(patch.resize((isz, isz), Image.LANCZOS), (ix, iy))
    ay = iy + isz // 2
    d.line([412, ay, 486, ay], fill=ARROW, width=4)
    d.polygon([(494, ay), (478, ay - 9), (478, ay + 9)], fill=ARROW)
    bx, max_w, _ = caption_layout(patch, "", st)
    y = draw_label(img, d, 530, 138, BOT_EMOJI, "Caption:", GREEN, st)
    draw_bubble(d, bx, y, wrap(" ".join(words), st.body, max_w), BOT_BUBBLE, st, cursor)
    return img


def qa_layout_bottom(q, a, st):
    max_w = BUBBLE_RIGHT - MARGIN_X - 2 * BUBBLE_PAD_X - st.cursor_w - 10
    y = 255 + EMOJI_SIZE + LABEL_GAP + bubble_size(wrap(q, st.body, max_w), st)[1]
    y += SECTION_GAP + EMOJI_SIZE + LABEL_GAP + bubble_size(wrap(a, st.body, max_w), st)[1]
    return y


def qa_frame(n, q_words, a_words, st, q_cursor, a_cursor):
    img, d = base_frame()
    max_w = BUBBLE_RIGHT - MARGIN_X - 2 * BUBBLE_PAD_X - st.cursor_w - 10
    d.text((MARGIN_X, 168), f"Q&A Round {n}", font=st.round, fill=GREY)
    y = draw_label(img, d, MARGIN_X, 255, USER_EMOJI, "User:", BLUE, st)
    y = draw_bubble(d, MARGIN_X, y, wrap(" ".join(q_words), st.body, max_w), USER_BUBBLE, st, q_cursor)
    if a_words is not None:
        y = draw_label(img, d, MARGIN_X, y + SECTION_GAP, BOT_EMOJI, "LLaVA-LE:", GREEN, st)
        draw_bubble(d, MARGIN_X, y, wrap(" ".join(a_words), st.body, max_w), BOT_BUBBLE, st, a_cursor)
    return img


def pick_style(caption, qa):
    for scale in (1.0, 0.92, 0.85, 0.78, 0.72, 0.66, 0.6):
        st = Style(scale)
        if (not caption or caption_layout(None, caption, st)[2] <= BOTTOM_LIMIT) and all(
            qa_layout_bottom(q, a, st) <= BOTTOM_LIMIT for q, a in qa
        ):
            return st, scale
    print("  warning: text still overflows at smallest font; shorten the caption/answers", file=sys.stderr)
    return st, scale


def build_frames(patch, caption, qa):
    st, scale = pick_style(caption, qa)
    if scale < 1:
        print(f"  font scaled to {scale:.0%} to fit")
    frames = []
    cw = caption.split() if caption else []
    for i in range(1, len(cw) + 1):
        done = i == len(cw)
        hold = T_CAPTION_HOLD if qa else T_FINAL_HOLD  # caption-only GIFs end on the long hold
        frames.append((caption_frame(patch, cw[:i], st, not done), hold if done else T_CAPTION_WORD))
    for r, (q, a) in enumerate(qa, 1):
        qw, aw = q.split(), a.split()
        for i in range(1, len(qw) + 1):
            done = i == len(qw)
            frames.append((qa_frame(r, qw[:i], None, st, not done, False), T_QUESTION_HOLD if done else T_QA_WORD))
        for i in range(1, len(aw) + 1):
            done = i == len(aw)
            hold = T_FINAL_HOLD if r == len(qa) else T_ROUND_HOLD
            frames.append((qa_frame(r, qw, aw[:i], st, False, not done), hold if done else T_QA_WORD))
    return frames


def save_gif(frames, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pal = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f, _ in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=[t for _, t in frames],
                loop=0, optimize=True, disposal=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--preview", action="store_true", help="also save hold frames as PNG next to each GIF")
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(args.config))
    cfg = json.load(open(args.config, encoding="utf-8"))
    rel = lambda p: p if os.path.isabs(p) else os.path.join(root, p)

    for ex in cfg["examples"]:
        rec = json.load(open(rel(ex["json"]), encoding="utf-8")) if "json" in ex else {}
        image = rel(ex.get("image") or os.path.splitext(ex["json"])[0] + ".png")
        caption = ex.get("caption") or rec["caption"]
        if not ex.get("show_caption", True):
            caption = ""
        elif ex.get("caption_sentences"):
            caption = first_sentences(caption, ex["caption_sentences"])
        qa = ex.get("qa") or [rec["qa"][i] for i in ex.get("qa_rounds", [0, 1])]
        out = rel(ex["out"])
        print(f"Rendering {os.path.relpath(out)} ({len(caption.split())}-word caption, {len(qa)} QA rounds)")

        frames = build_frames(Image.open(image).convert("RGB"), caption, qa)
        save_gif(frames, out)
        total = sum(t for _, t in frames) / 1000
        print(f"  {len(frames)} frames, {total:.1f}s, {os.path.getsize(out) / 1e6:.2f} MB")
        if args.preview:
            stem = os.path.splitext(out)[0]
            for k, (f, t) in enumerate(x for x in frames if x[1] >= T_QUESTION_HOLD):
                f.save(f"{stem}_preview_{k}.png")


if __name__ == "__main__":
    main()
