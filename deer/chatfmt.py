"""
Chat format for training text.

Datasets are written the readable way:

    User: How do I make a player jump in Unity?
    Deer: Add a Rigidbody2D and apply an upward force:
    ```csharp
    if (Input.GetKeyDown(KeyCode.Space)) rb.AddForce(Vector2.up * jumpForce, ForceMode2D.Impulse);
    ```

A reply keeps going over as many lines as it needs (code included) until the next line that
starts with "User:" outside a ``` code block. For training, every reply is closed with an
invisible end-of-reply byte (EOT) so the model learns where its answer stops.
"""
import re

EOT = "\x03"
EOT_BYTE = 3
_FENCE = re.compile(r"^\s*```")


def parse(text):
    """Split text into units: {"q": question, "a": answer} pairs (a is None when unanswered),
    or {"plain": text} blocks for anything that isn't a conversation (docs, code files)."""
    units, buf = [], []
    state = {"q": None, "mode": None}
    in_code = False

    def end_block():
        body = "\n".join(buf)
        if state["mode"] == "q":
            state["q"] = body.strip()
        elif state["mode"] == "a":
            units.append({"q": state["q"], "a": body.strip("\n").rstrip()})
            state["q"] = None
        elif state["mode"] == "plain" and body.strip():
            units.append({"plain": body.strip("\n")})
        buf.clear()
        state["mode"] = None

    def close_question():
        if state["q"] is not None:
            units.append({"q": state["q"], "a": None})
            state["q"] = None

    text = text.replace("\r\n", "\n").replace(EOT, "\n" + EOT + "\n")
    for line in text.split("\n"):
        if line == EOT:  # end of a reply in text that was already converted for training
            end_block()
            in_code = False
            continue
        if not in_code and line.startswith("User:"):
            end_block()
            close_question()
            state["mode"] = "q"
            rest = line[5:].lstrip()
            buf.append(rest)
            in_code = bool(_FENCE.match(rest)) and rest.count("```") % 2 == 1  # "User: ```python" opens a block
            continue
        if not in_code and line.startswith("Deer:") and state["mode"] == "q":
            end_block()
            state["mode"] = "a"
            rest = line[5:].lstrip()
            buf.append(rest)
            in_code = bool(_FENCE.match(rest)) and rest.count("```") % 2 == 1  # "Deer: ```python" opens a block
            continue
        if _FENCE.match(line):
            in_code = not in_code
        if state["mode"] is None:
            state["mode"] = "plain"
        buf.append(line)
    end_block()
    close_question()
    return units


def unit_text(u, for_training=False):
    """Readable text for one unit (with the end-of-reply byte when for_training)."""
    if "plain" in u:
        return u["plain"]
    if u.get("a") is None:
        return f"User: {u['q']}"
    return f"User: {u['q']}\nDeer: {u['a']}" + (EOT if for_training else "")


def to_training(text):
    return "\n".join(unit_text(u, True) for u in parse(text)) + "\n"


def lesson(prompt, answer):
    """One taught Q&A in training form (answer may be several lines, e.g. code)."""
    return f"User: {prompt.strip()}\nDeer: {answer.strip()}{EOT}\n"
