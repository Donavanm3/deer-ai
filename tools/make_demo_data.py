"""
Generates starter datasets so you can test the full pipeline before you have
your own data. Nothing here is a pre-made AI: it's procedural drawing and
templated text.

  python tools/make_demo_data.py            -> examples/shapes.zip, examples/deer_chat.txt
"""
import io
import os
import random
import zipfile
from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")

SHAPES = {
    "red circle": ((220, 50, 50), "circle"),
    "blue square": ((50, 90, 220), "square"),
    "green triangle": ((40, 170, 70), "triangle"),
    "orange ring": ((240, 140, 30), "ring"),
}


def draw_shape(kind, color, size=64, rnd=random):
    bg = tuple(rnd.randint(215, 255) for _ in range(3))
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    r = rnd.randint(size // 5, size // 3)
    cx, cy = rnd.randint(r + 2, size - r - 2), rnd.randint(r + 2, size - r - 2)
    col = tuple(max(0, min(255, c + rnd.randint(-25, 25))) for c in color)
    if kind == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif kind == "ring":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=max(3, r // 3))
    elif kind == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=col)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=col)
    return im


def make_shapes(per_class=300, seed=0):
    rnd = random.Random(seed)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, (color, kind) in SHAPES.items():
            for i in range(per_class):
                b = io.BytesIO()
                draw_shape(kind, color, rnd=rnd).save(b, "PNG")
                z.writestr(f"shapes/{name}/{i:04d}.png", b.getvalue())
    return buf.getvalue()


GREET_U = ["Hello!", "Hi", "Hey there", "Good morning", "Hi Deer", "Hello Deer AI", "yo", "Hey!"]
GREET_A = ["Hello! How can I help you today?", "Hi there! What would you like to do?",
           "Hey! I'm Deer AI. What can I do for you?", "Hello! Ask me anything, or send me a picture."]
QA = [
    (["What is your name?", "Who are you?", "what are you"],
     ["I'm Deer AI, a model built and trained from scratch.", "My name is Deer AI."]),
    (["Who made you?", "Who built you?", "who created you"],
     ["I was built by Dashlith and trained on my own data.", "Dashlith made me, one training step at a time."]),
    (["What can you do?", "help", "What are your skills?"],
     ["I can chat, look at pictures and tell you what I see, and draw new images.",
      "Send me a picture to analyze, or ask me to imagine something."]),
    (["How are you?", "How's it going?", "how are you doing"],
     ["I'm doing well, thanks for asking!", "Running smoothly. How about you?"]),
    (["Can you draw?", "Make me a picture", "Can you create images?"],
     ["Yes! Switch on Imagine and describe what you want.", "I can draw the subjects I was trained on. Try Imagine."]),
    (["Thanks!", "Thank you", "thx"], ["You're welcome!", "Anytime!", "Happy to help."]),
    (["Bye", "Goodbye", "See you later"], ["Goodbye! Come back soon.", "See you later!"]),
    (["What is a deer?", "Tell me about deer"],
     ["A deer is a hoofed animal. Males usually grow antlers every year.",
      "Deer live in forests and fields and eat plants, leaves and acorns."]),
    (["What is 2 plus 2?", "2+2"], ["2 plus 2 is 4."]),
    (["What colors do you know?", "name a color"], ["Red, blue, green, orange, and many more."]),
]


def make_chat(n=6000, seed=0):
    rnd = random.Random(seed)
    lines = []
    for _ in range(n):
        if rnd.random() < 0.3:
            lines.append(f"User: {rnd.choice(GREET_U)}\nDeer: {rnd.choice(GREET_A)}\n")
        else:
            qs, ans = rnd.choice(QA)
            lines.append(f"User: {rnd.choice(qs)}\nDeer: {rnd.choice(ans)}\n")
    return "".join(lines)



# ========================================================= coding + game dev
# Original, hand-written examples with small variations so a model can learn the patterns.
# Replies span several lines and use ``` code blocks (Deer AI's chat format supports that).
CODE_TOPICS = [
    (["How do I make a game loop in JavaScript?", "Write a JS game loop", "game loop with requestAnimationFrame"],
     "Use requestAnimationFrame and a time step so the game runs the same speed on every screen:\n```javascript\n"
     "let last = performance.now();\nfunction loop(now) {\n  const dt = (now - last) / 1000;\n  last = now;\n"
     "  update(dt);\n  draw();\n  requestAnimationFrame(loop);\n}\nrequestAnimationFrame(loop);\n```"),
    (["How do I move a player with the arrow keys in JavaScript?", "keyboard movement in a canvas game", "WASD movement JS"],
     "Track which keys are held, then move in your update step:\n```javascript\nconst keys = {};\n"
     "addEventListener(\"keydown\", e => keys[e.key] = true);\naddEventListener(\"keyup\", e => keys[e.key] = false);\n"
     "function update(dt) {\n  const speed = {speed};\n  if (keys.ArrowLeft || keys.a) player.x -= speed * dt;\n"
     "  if (keys.ArrowRight || keys.d) player.x += speed * dt;\n  if (keys.ArrowUp || keys.w) player.y -= speed * dt;\n"
     "  if (keys.ArrowDown || keys.s) player.y += speed * dt;\n}\n```"),
    (["How do I check collision between two rectangles?", "AABB collision code", "box collision in a 2D game"],
     "Two boxes overlap when they overlap on both axes:\n```javascript\nfunction hits(a, b) {\n"
     "  return a.x < b.x + b.w && a.x + a.w > b.x &&\n         a.y < b.y + b.h && a.y + a.h > b.y;\n}\n```"),
    (["Make a tiny HTML game", "Write a simple browser game", "HTML canvas game example"],
     "Here's a small game: move the square with the arrow keys and collect the coin.\n```html\n<!doctype html>\n"
     "<canvas id=\"c\" width=\"480\" height=\"320\"></canvas>\n<script>\nconst g = document.getElementById(\"c\").getContext(\"2d\");\n"
     "const p = {x: 40, y: 40, s: 20}, coin = {x: 300, y: 200};\nlet score = 0; const keys = {};\n"
     "onkeydown = e => keys[e.key] = true; onkeyup = e => keys[e.key] = false;\nfunction frame() {\n"
     "  if (keys.ArrowLeft) p.x -= {step}; if (keys.ArrowRight) p.x += {step};\n  if (keys.ArrowUp) p.y -= {step}; if (keys.ArrowDown) p.y += {step};\n"
     "  if (Math.abs(p.x - coin.x) < 20 && Math.abs(p.y - coin.y) < 20) {\n    score++; coin.x = Math.random() * 440 + 20; coin.y = Math.random() * 280 + 20;\n  }\n"
     "  g.fillStyle = \"#1f2b25\"; g.fillRect(0, 0, 480, 320);\n  g.fillStyle = \"#c9955b\"; g.fillRect(p.x, p.y, p.s, p.s);\n"
     "  g.fillStyle = \"gold\"; g.beginPath(); g.arc(coin.x, coin.y, 8, 0, 7); g.fill();\n"
     "  g.fillStyle = \"white\"; g.fillText(\"Score: \" + score, 10, 20);\n  requestAnimationFrame(frame);\n}\nframe();\n</script>\n```"),
    (["How do I make a player jump in Unity?", "Unity 2D jump script", "jumping in Unity C#"],
     "Add a Rigidbody2D and use an impulse when the player is on the ground:\n```csharp\nusing UnityEngine;\n\n"
     "public class PlayerJump : MonoBehaviour\n{\n    public float jumpForce = {jump}f;\n    Rigidbody2D rb;\n    bool grounded;\n\n"
     "    void Awake() { rb = GetComponent<Rigidbody2D>(); }\n\n    void Update()\n    {\n"
     "        if (grounded && Input.GetKeyDown(KeyCode.Space))\n            rb.AddForce(Vector2.up * jumpForce, ForceMode2D.Impulse);\n    }\n\n"
     "    void OnCollisionEnter2D(Collision2D c) { grounded = true; }\n    void OnCollisionExit2D(Collision2D c) { grounded = false; }\n}\n```"),
    (["How do I move a character in Unity?", "Unity player movement script", "Unity C# movement"],
     "Read the input axes and move with the Rigidbody so physics stays smooth:\n```csharp\nusing UnityEngine;\n\n"
     "public class PlayerMove : MonoBehaviour\n{\n    public float speed = {uspeed}f;\n    Rigidbody2D rb;\n\n"
     "    void Awake() { rb = GetComponent<Rigidbody2D>(); }\n\n    void FixedUpdate()\n    {\n"
     "        var input = new Vector2(Input.GetAxisRaw(\"Horizontal\"), Input.GetAxisRaw(\"Vertical\"));\n"
     "        rb.velocity = input.normalized * speed;\n    }\n}\n```"),
    (["How do I spawn enemies every few seconds in Unity?", "Unity enemy spawner", "spawn prefabs on a timer"],
     "Use a coroutine that waits, then instantiates a prefab at a random spawn point:\n```csharp\nusing System.Collections;\nusing UnityEngine;\n\n"
     "public class Spawner : MonoBehaviour\n{\n    public GameObject enemy;\n    public Transform[] points;\n    public float every = {every}f;\n\n"
     "    IEnumerator Start()\n    {\n        while (true)\n        {\n            var p = points[Random.Range(0, points.Length)];\n"
     "            Instantiate(enemy, p.position, Quaternion.identity);\n            yield return new WaitForSeconds(every);\n        }\n    }\n}\n```"),
    (["How do I collect coins in Unity?", "coin pickup Unity", "Unity trigger pickup"],
     "Make the coin's collider a trigger and count it when the player touches it:\n```csharp\nusing UnityEngine;\n\n"
     "public class Coin : MonoBehaviour\n{\n    public static int count;\n\n    void OnTriggerEnter2D(Collider2D other)\n    {\n"
     "        if (!other.CompareTag(\"Player\")) return;\n        count++;\n        Destroy(gameObject);\n    }\n}\n```"),
    (["How do I reverse a string in Python?", "python reverse string", "reverse text in python"],
     "Slice it with a step of -1:\n```python\ndef reverse(text):\n    return text[::-1]\n\nprint(reverse(\"{word}\"))\n```"),
    (["Write FizzBuzz in Python", "python fizzbuzz", "FizzBuzz example"],
     "Check 15 first, then 3 and 5:\n```python\nfor i in range(1, {n}):\n    if i % 15 == 0:\n        print(\"FizzBuzz\")\n"
     "    elif i % 3 == 0:\n        print(\"Fizz\")\n    elif i % 5 == 0:\n        print(\"Buzz\")\n    else:\n        print(i)\n```"),
    (["How do I read a file in Python?", "python read text file", "open a file in python"],
     "Use a with block so the file always closes:\n```python\nwith open(\"{file}\", encoding=\"utf-8\") as f:\n"
     "    for line in f:\n        print(line.rstrip())\n```"),
    (["Make a Python class for a game character", "python player class", "class example python game"],
     "A small class with health and damage:\n```python\nclass Character:\n    def __init__(self, name, health={hp}):\n"
     "        self.name = name\n        self.health = health\n\n    def hit(self, amount):\n        self.health = max(0, self.health - amount)\n"
     "        return self.health == 0\n\nhero = Character(\"Deer\")\nprint(hero.hit(10), hero.health)\n```"),
    (["How do I make a simple game in Pygame?", "pygame starter", "pygame window and player"],
     "Open a window, move a square, and cap the frame rate:\n```python\nimport pygame\n\npygame.init()\nscreen = pygame.display.set_mode((640, 480))\n"
     "clock = pygame.time.Clock()\nx, y = 320, 240\nrunning = True\nwhile running:\n    for e in pygame.event.get():\n"
     "        if e.type == pygame.QUIT:\n            running = False\n    keys = pygame.key.get_pressed()\n"
     "    x += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * {step}\n    y += (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * {step}\n"
     "    screen.fill((31, 43, 37))\n    pygame.draw.rect(screen, (201, 149, 91), (x, y, 24, 24))\n    pygame.display.flip()\n    clock.tick(60)\npygame.quit()\n```"),
    (["How do I move a player in Godot?", "Godot 4 movement script", "GDScript player movement"],
     "Use CharacterBody2D and move_and_slide:\n```gdscript\nextends CharacterBody2D\n\n@export var speed := {uspeed}0.0\n\n"
     "func _physics_process(delta):\n\tvar dir := Input.get_vector(\"ui_left\", \"ui_right\", \"ui_up\", \"ui_down\")\n"
     "\tvelocity = dir * speed\n\tmove_and_slide()\n```"),
    (["How do I move a player in LOVE2D?", "love2d movement", "Lua game player movement"],
     "Update position in love.update using dt:\n```lua\nlocal player = { x = 100, y = 100, speed = {speed} }\n\n"
     "function love.update(dt)\n  if love.keyboard.isDown(\"right\") then player.x = player.x + player.speed * dt end\n"
     "  if love.keyboard.isDown(\"left\") then player.x = player.x - player.speed * dt end\nend\n\n"
     "function love.draw()\n  love.graphics.rectangle(\"fill\", player.x, player.y, 24, 24)\nend\n```"),
    (["What is a game loop?", "explain the game loop", "how do games update each frame?"],
     "A game loop runs every frame: read input, update the game (movement, physics, AI) using the time since the last "
     "frame, then draw everything. Using that time step keeps the game the same speed on fast and slow computers."),
    (["What is delta time?", "why multiply by dt?", "explain delta time"],
     "Delta time is how long the last frame took. Multiplying speeds by it makes movement depend on real time instead of "
     "frame rate, so a player moves {speed} pixels per second whether the game runs at 30 or 144 FPS."),
    (["Hello!", "Hi", "Hey Deer"], "Hi! I'm Deer AI. I can help you write code and make games. What are you building?"),
    (["What can you do?", "help", "What are you good at?"],
     "I help with coding and game creation: game loops, movement, collisions, Unity, Godot, Pygame and web games. "
     "Turn on Imagine to draw game art like sprites."),
]


def make_code_chat(n=3000, seed=0):
    rnd = random.Random(seed)
    words = ["deer", "forest", "acorn", "pixel", "sprite"]
    out = []
    for _ in range(n):
        qs, a = rnd.choice(CODE_TOPICS)
        vals = {"speed": rnd.choice([120, 160, 200, 240]), "step": rnd.choice([3, 4, 5]), "jump": rnd.choice([6, 7, 8]),
                "uspeed": rnd.choice([4, 5, 6]), "every": rnd.choice([2, 3, 5]), "word": rnd.choice(words),
                "n": rnd.choice([16, 31, 51]), "file": rnd.choice(["notes.txt", "level1.txt", "save.txt"]),
                "hp": rnd.choice([50, 100, 120])}
        ans = a
        for k, v in vals.items():
            ans = ans.replace("{" + k + "}", str(v))
        out.append(f"User: {rnd.choice(qs)}\nDeer: {ans}\n")
    return "".join(out)


# pixel-art game sprites, drawn on a 16x16 grid and scaled up 2x
def _sprite(kind, rnd):
    from PIL import ImageDraw
    im = Image.new("RGB", (16, 16), tuple(rnd.randint(200, 240) for _ in range(3)))
    d = ImageDraw.Draw(im)
    j = lambda c: tuple(max(0, min(255, v + rnd.randint(-20, 20))) for v in c)
    ox, oy = rnd.randint(-1, 1), rnd.randint(-1, 1)
    o = lambda pts: [(x + ox, y + oy) for x, y in pts]
    if kind == "coin":
        d.ellipse(o([(3, 3), (12, 12)]), fill=j((235, 190, 40)), outline=j((160, 110, 20)))
        d.line(o([(6, 5), (6, 9)]), fill=(255, 245, 190))
    elif kind == "heart":
        d.polygon(o([(8, 13), (2, 7), (2, 4), (4, 2), (6, 2), (8, 4), (10, 2), (12, 2), (14, 4), (14, 7)]), fill=j((220, 40, 60)))
        d.point(o([(4, 4)]), fill=(255, 200, 200))
    elif kind == "slime":
        d.ellipse(o([(2, 6), (13, 14)]), fill=j((80, 190, 70)))
        d.rectangle(o([(2, 11), (13, 14)]), fill=j((80, 190, 70)))
        d.point(o([(5, 9), (10, 9)]), fill=(20, 20, 20))
    elif kind == "sword":
        d.line(o([(3, 12), (12, 3)]), fill=j((200, 205, 215)), width=2)
        d.line(o([(3, 9), (6, 12)]), fill=j((120, 80, 40)), width=2)
        d.point(o([(2, 13)]), fill=j((120, 80, 40)))
    elif kind == "tree":
        d.rectangle(o([(7, 10), (8, 14)]), fill=j((110, 70, 35)))
        d.ellipse(o([(3, 2), (12, 11)]), fill=j((40, 140, 60)))
    elif kind == "potion":
        col = j(rnd.choice([(160, 60, 200), (60, 120, 220), (220, 70, 70)]))
        d.rectangle(o([(7, 2), (8, 5)]), fill=j((180, 200, 210)))
        d.ellipse(o([(4, 5), (11, 13)]), fill=col, outline=j((60, 60, 80)))
    elif kind == "key":
        d.ellipse(o([(2, 5), (7, 10)]), outline=j((225, 185, 50)), width=2)
        d.line(o([(7, 7), (13, 7)]), fill=j((225, 185, 50)), width=2)
        d.line(o([(11, 7), (11, 10)]), fill=j((225, 185, 50)))
    elif kind == "gem":
        d.polygon(o([(8, 2), (13, 6), (8, 14), (3, 6)]), fill=j((60, 200, 220)), outline=j((20, 110, 140)))
        d.line(o([(3, 6), (13, 6)]), fill=j((150, 235, 245)))
    return im.resize((32, 32), Image.NEAREST)


SPRITES = ["coin", "heart", "slime", "sword", "tree", "potion", "key", "gem"]


def make_sprites(per_class=150, seed=0):
    rnd = random.Random(seed)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in SPRITES:
            for i in range(per_class):
                b = io.BytesIO()
                _sprite(name, rnd).save(b, "PNG")
                z.writestr(f"sprites/{name}/{i:04d}.png", b.getvalue())
    return buf.getvalue()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = {"shapes.zip": make_shapes(), "sprites.zip": make_sprites(),
             "deer_chat.txt": make_chat().encode("utf-8"), "code_chat.txt": make_code_chat().encode("utf-8")}
    for name, blob in files.items():
        with open(os.path.join(OUT, name), "wb") as f:
            f.write(blob)
    print("Wrote examples/" + ", examples/".join(files) + ". These are optional starter data; you still train the models.")
