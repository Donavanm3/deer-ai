# Deer AI

A self-hosted AI platform where every model is built and trained from scratch.
No pretrained weights, no AI frameworks (PyTorch, TensorFlow), no AI APIs.
The neural-network engine in `deer/engine/` is written by hand on top of NumPy,
with a hand-derived gradient for every operation.

## What it's for

Deer AI focuses on **coding and game creation**, and still does general text and images:

- **Chat** answers can span many lines, so the model can write whole code snippets. Code shows
  in blocks with **Copy**, and HTML or JavaScript blocks have **Run**, which plays the game right
  in the chat inside a sealed-off frame (it can't touch your account or the page).
- **Image creation** is geared to game art: every generated picture has **Pixel art** (snaps to a
  small palette), **No background** (see-through PNG) and **Download**, plus **Download all as a
  sprite sheet**.
- **Nothing is pre-trained.** Every model starts empty and learns only from data you give it
  and the training you run. `tools/make_demo_data.py` can write optional starter data
  (coding/game-dev conversations, game sprites) so you can try it out; you still train on it.

## Quick start

```bash
pip install -r requirements.txt
python tools/make_demo_data.py     # optional starter datasets in examples/
python run.py                      # http://127.0.0.1:8000
```

The **first account you create becomes the admin**. Open **Admin** from the chat sidebar.

To reach it from other devices: `python run.py --host 0.0.0.0 --port 8000`
(put it behind HTTPS, e.g. Caddy or nginx, before exposing it to the internet).

## Setting up and updating from a phone

Everything can be done from a phone; no computer needed.

**A new VPS.** Install an SSH app: **Termius** (iPhone and Android) or **Termux** (Android).
1. Download `deer-ai.zip` on the phone.
2. Upload it to the server: in Termius open the host's **SFTP** view and upload it to `/root`
   (Termux: `scp ~/storage/downloads/deer-ai.zip root@YOUR_IP:/root/`).
3. In the SSH terminal, one command installs everything (packages, service, firewall, HTTPS):
   ```
   cd /root && unzip -o deer-ai.zip && bash deer-ai/tools/install_vps.sh ai.yourdomain.com
   ```
   It's safe to run again later; it also repairs a broken setup.
4. Open `https://ai.yourdomain.com` and create your account first (it becomes the admin).

**Updates (Admin > Updates).**
- **Upload a zip:** pick the new `deer-ai.zip` from your phone's files and press Update. Your data,
  models, users and settings stay. The current version is backed up first; the last 3 backups can
  be restored with one tap. If a new version can't start, Deer AI puts the previous one back by
  itself and tells you why on this page.
- **GitHub:** enter your repo (make an empty private one on github.com first) and press Connect.
  The server makes its own key; copy it into the repo's **Deploy keys** (tick *Allow write access*)
  using the link on the page. Then **Push this server's code to GitHub** once, and **Update from
  GitHub** whenever the code there changes. Optionally add the webhook shown on the page and switch
  on automatic updates: every push to your branch updates the server (after any running training).
- Updates wait while a model is training.

Existing servers need `apt install -y git openssh-client` once for the GitHub buttons (the
installer does it); zip updates work without it.

## Hosting with GitHub

GitHub can't run Deer AI's server (it needs Python, a database and CPU for training), so the server
stays on your VPS. GitHub helps in three ways; use any of them:

**1. Auto-deploy (recommended).** Keep the code in a GitHub repo. Every push to `main` runs all the
tests on GitHub, and only if they pass, uploads the new code to your VPS and restarts Deer AI
(`.github/workflows/deploy.yml`). Your data (`deer_data`), settings (`deer.env`) and `venv` on the
server are never touched. If Deer AI doesn't come back up, the run fails and shows the server log.
Set up once in the repo's Settings > Secrets and variables > Actions:
- Secrets: `VPS_HOST` (your server's IP), `VPS_USER` (`root`), `VPS_SSH_KEY` (a private key the server accepts),
  optionally `VPS_HOST_KEY` (output of `ssh-keyscan YOUR_IP`) to pin the server's identity.
- Variable: `DEPLOY_TO_VPS` = `true`.

**2. The website on GitHub Pages.** The chat, admin, plans and API pages are published as static
files at `https://YOURNAME.github.io/REPO/` and talk to your VPS for everything else
(`.github/workflows/pages.yml`, built by `tools/build_pages.py`).
- Repo Settings > Pages > Source: **GitHub Actions**.
- Variable `DEER_API_URL` = your server's https address, e.g. `https://ai.dashlith.com`.
- On the server, add to `/opt/deer-ai/deer.env`: `DEER_ALLOWED_ORIGINS=https://YOURNAME.github.io`
  and restart. Only listed sites may call the server from a browser.
- A custom domain can be set in Settings > Pages. Sign-ins are per site, so people sign in once on
  the Pages address. Your VPS address keeps working too.

**3. Codespaces.** On the repo page: Code > Codespaces > Create. It installs everything, writes the
starter data and starts Deer AI; the page opens in your browser (phone included). It's private to
you and sleeps when idle, so it's for trying changes, not for your users.

## Run it on a phone

**Install the app icon (any phone).** Open your Deer AI address in Chrome (Android) and
choose *Add to Home screen*, or in Safari (iPhone) tap Share, then *Add to Home Screen*.
It opens full screen like an app. Needs HTTPS or the phone's own `localhost`.

**Host the server on an Android phone (Termux).** iPhones can't run the server.
1. Install **Termux** from F-Droid (the Play Store version is outdated).
2. Copy `deer-ai.zip` to the phone, then in Termux:
   ```bash
   termux-setup-storage
   pkg install -y unzip
   cp ~/storage/downloads/deer-ai.zip ~ && cd ~ && unzip deer-ai.zip && cd deer-ai
   bash tools/termux_setup.sh      # 10-20 min the first time
   bash tools/termux_start.sh      # then open http://127.0.0.1:8000 in Chrome
   ```
3. `bash tools/termux_start.sh public` also prints a temporary `https://….trycloudflare.com`
   link so others can reach it. That link changes every restart; for a permanent address
   (needed for PayPal webhooks) use a named Cloudflare Tunnel on your own domain.
4. In Android settings, set Termux battery usage to *Unrestricted* so it isn't killed.
   If training won't start, add `DEER_TRAIN_MODE=thread` to `deer.env`.

## For users

- Replies stream in word by word. A reply is capped at 20 seconds and 300 characters
  (`DEER_REPLY_SECONDS`, `DEER_MAX_REPLY`), so the chat never hangs on "Thinking".
- A usage bar under the message box shows how many messages (or pictures in Imagine mode)
  are left and when more free up. The **Plans** page shows all three meters.
- **Save** under any reply, analysis or picture keeps it in **Saved** (sidebar).
  Pictures have a Download link; the arrow at the top exports the whole chat as Markdown.
- Every chat is saved automatically and listed in the sidebar.

## Testing as the admin

Open **Plans** (or "Test as user" in the sidebar) and switch on **Test mode**. You then see only
published models and hit the same limits as users. Pick "My real plan" to try PayPal sandbox
checkout, or preview any plan without paying. **Reset my usage** clears your test counts.
The admin panel keeps working while test mode is on.

## Building better datasets (Admin > Build dataset)

Combine any datasets, including the lessons users taught them, into a new cleaner one:

- **Text:** choose original text and/or user lessons, then remove duplicates, junk and empty
  answers, keep only User/Deer pairs, set length limits, drop anything containing words you list
  (swear words, spam), or keep only what mentions certain words. Cap or shuffle the result.
- **Images:** merge datasets, leave classes out, rename them (two classes with the same name
  merge), drop duplicate pictures (fingerprint match), leave out user-taught pictures, set a
  minimum size, drop tiny classes and balance the rest with a max per class.
- **Preview** shows how much is kept, what was removed and why, and samples. **Create dataset**
  saves it (the original datasets are never changed) and suggests a model size for that much data;
  **Train a model on it** fills in the new-model form.

The new dataset records where it came from and which filters built it.

## Models that train models

- **Start from a model** (new-model form): the new model copies what a trained model of the same
  type already knows, so it starts far ahead. A bigger model inherits every part that fits
  (for example the first layers of a deeper chat model); the extra parts start fresh. Image
  models keep what they know about classes the new dataset shares.
- **Teacher** (chat and image analysis): while training, the new model learns from the data and
  from the teacher's full answers (knowledge distillation). The slider sets how much it follows
  the teacher. Use it to make a small fast model that behaves like a big one, or to guide a
  bigger model early on. A chat teacher must read at least as much context (block_size).
- **Models as data sources** (Build dataset): a chat model can write replies to the questions in
  your data; an analysis model can label pictures in an "unlabeled" folder (upload a zip with no
  folders), keeping only the ones it's sure about; a creation model can draw extra pictures per
  class for analysis models. All of it goes through the same filters and preview.

Honest limits: a model can't teach more than it knows. Starting from a model and teachers save
time and let knowledge move between sizes; real gains in smarts still come from more good data.

## Learning from users over time

**What changed in how lessons are learned:** approved lessons are now mixed into every training
batch (25% normally, 50% in learning runs) instead of being a drop in the ocean of the base text.
In testing, a model that recalled 0 of 5 taught answers before now recalls them word for word,
while its general quality stays level. Learning runs report how well it knows its lessons
before and after, next to the usual held-out score.

**Learning from what users do** (Admin > Learning, on by default):
- *Learn from what users write*: their messages are added to the chat model's training text.
- *Learn from pictures users send*: when the analysis model is at least 90% sure, the picture
  arrives in the Learning inbox labeled with its answer (or goes straight in with auto-approve).
- Users can opt out on their **Plans** page ("Let Deer AI learn from my chats and pictures").
  Opted-out users' chats and pictures are never used. The chat shows a short notice with a link.
- **Build dataset** can also pull straight from users: all feedback (reviewed or not), everything
  users wrote, whole conversations minus disliked replies, and pictures users sent (labeled by
  their Yes/No feedback; the rest can be labeled by an analysis model).

### Feedback and auto-learn

Deer AI gets better the more people use it:

- Under each chat reply users can tap thumbs up or down, and write a better answer.
- Under an image analysis they answer "Is this right?" and can pick or type the correct label.
  A brand-new label becomes a new class; analysis and creation models grow into it
  without forgetting what they already know.
- Feedback lands in **Admin > Learning**. Approve or reject it (or switch on auto-approve).
  Approved chat lessons go into the dataset's `learned.txt`; approved images go into its folders.
- On a model's page, **Learn from users** turns on auto-learn: once enough new lessons arrive,
  the model fine-tunes itself with a gentle learning rate. Every run backs up the previous
  checkpoint, is scored on held-out data, and is undone automatically if the model got worse.
  **Learn now** runs one immediately; **Undo last training run** restores the backup.
- Ratings on generated images are kept as stats only. The generator is never trained on its
  own output, which would make it worse over time.

The scheduler checks every 5 minutes (`DEER_LEARN_INTERVAL`, in seconds).

## Developer API (API page, Admin > API)

Anyone with an account can call Deer AI's models from their own games, bots and apps.

- **Keys:** create keys on the **API** page (linked from the chat sidebar). A key is shown once,
  then only its first characters; it's stored hashed. Each key has permissions (chat / analyze /
  create pictures), a requests-per-minute limit, an optional monthly spending limit, and can be
  revoked instantly.
- **Pay as you go:** users buy prepaid credit with PayPal (one-time payment, credit never expires).
  Each request's cost is taken from their balance: chat per 1,000 tokens in and out (1 token = 1
  byte for Deer models), pictures per image. A request is refused up front if it could cost more
  than the balance or the key's monthly limit, so nobody goes below zero. Payments are credited
  exactly once, even if the buyer double-clicks or PayPal's webhook arrives too.
- **Usage:** a daily chart (hover for details), cost per endpoint, token and picture totals,
  recent requests and credit history. Every response also returns its `usage`, `cost` and the new
  `balance`.
- **Admin > API:** switch the API on or off, set prices, a welcome credit for new keys, the
  top-up amounts, give or take credit from any user, and see usage, revenue and credit held.
  The owner's own keys are always free.
- **Endpoints:** `POST /v1/chat` (optionally streaming), `POST /v1/analyze`, `POST /v1/imagine`,
  `GET /v1/models`, `GET /v1/usage`. Examples in curl, Python, JavaScript and Unity C# are on the
  API page. API requests are never used for training.

For top-ups, add `PAYMENT.CAPTURE.COMPLETED` to your PayPal webhook's events.

## Plans and PayPal

Signed-in users get the Free plan; paid plans raise their limits, like Claude's plans.
Messages count over a rolling window (5 hours by default), images and analyses per 24 hours.
Admins are unlimited. Edit names, prices and limits in **Admin > Plans**.

1. At developer.paypal.com, create an app under *Apps & Credentials* (Sandbox first).
2. Admin > Plans: paste the Client ID and Secret, choose Sandbox, press *Save and test*.
3. Press *Link to PayPal* on each paid plan. This creates the PayPal product and plan.
4. In the PayPal app settings, add a webhook with the URL shown in Admin > Plans and the
   listed events, then paste its **Webhook ID** and save again. Webhooks need public HTTPS.
5. Sign in as a regular user, open **Plans**, and subscribe with a sandbox buyer account.
6. When it works, repeat with **Live** credentials, relink the plans, and add a live webhook.

Every subscription is re-checked with PayPal on the server (the browser is never trusted),
webhooks are signature-verified, switching plans cancels the old subscription, and a
cancelled plan keeps working until the end of the paid month.

## Training two models at once

Up to two models train at the same time (`DEER_MAX_JOBS`, default 2), but the second one only
starts when the server can take it:

- the server has at least 2 CPU cores,
- other work on the machine (website visitors, game servers, anything else) isn't using more than
  half the cores (`DEER_BUSY_SHARE`, default 0.5), and
- at least 1.5 GB of memory is free (`DEER_MIN_FREE_MB`, default 1500).

Otherwise it **waits in line** and starts by itself as soon as there's room; nothing is refused.
The model page shows its place in line and why it's waiting, and **Remove from queue** takes it
out. Jobs that run together split the CPU cores between them instead of fighting over them, and
they re-balance when one finishes. Training also runs at low priority, so chat stays responsive.
Home shows the slots in use, CPU cores, load and free memory.

## Training data for coding

Text datasets accept plain text, code files (Python, JavaScript/TypeScript, HTML/CSS, C#, Lua,
GDScript, Java, C/C++, Rust, Go, shaders, mcfunction and more), or a **.zip of a whole project**.
Project zips skip art, build and dependency folders (node_modules, Library, .git…) and keep each
file with its path, so the model learns how real projects are laid out.

For question-and-answer training, write conversations like this. A reply runs until the next line
that starts with `User:` (outside a code block), so it can hold code:

    User: How do I make a player jump in Unity?
    Deer: Use an impulse on the Rigidbody2D:
    ```csharp
    rb.AddForce(Vector2.up * 7f, ForceMode2D.Impulse);
    ```

Put any plain text before the conversations; text right after a reply counts as part of it.
Behind the scenes each reply gets an invisible end marker so the model learns where to stop.

**Sizes:** chat models have *Code, small* and *Code, medium* sizes that read 256 and 384 characters
of context (code needs to see further back). They train slower on a CPU; a VPS with more cores
helps a lot. In testing, a small model trained for about 4 minutes on the starter coding data
answered with correct multi-line code and scored 90 on its held-out benchmark.

## Benchmarks (Admin > Benchmarks)

A benchmark is a fixed test; run any models on it and they're ranked on a leaderboard.

- **Chat:** questions from the held-out part of a dataset, or ones you write with the answers you
  expect. Scored on word overlap with the expected answer, exact answers, whether code in the
  reply is well-formed (Python is fully parsed; other languages checked for balanced brackets),
  and speed. See every question, expected answer and actual reply.
- **Image analysis:** a labeled image dataset. Accuracy overall and per class, the most common
  mix-ups, confidence and speed.
- **Image creation:** each class is drawn and an analysis model you pick judges whether it
  recognizes what was asked, plus variety and speed.

Test items are frozen when the benchmark is made, so scores stay comparable over time.

### Comparing with other AIs

Chat benchmarks can include outside models, so you can see how your models stack up:

- **Other AI models** (Benchmarks page): add ChatGPT (OpenAI), Claude (Anthropic), Gemini,
  OpenRouter, any OpenAI-compatible service, or a model running on your own server with Ollama.
  Enter the model's ID and your API key (not needed for Ollama), press **Add and test**, then tick
  it on a chat benchmark and press **Run**. They show on the leaderboard marked *other AI*.
- **Answer sheet** (no API key needed): download the questions, ask them in any AI's own app, fill in
  each answer after its `Deer:` line, and paste the sheet back. Marked *pasted*.
- To keep scoring fair, each question is sent with two other questions from the same benchmark as
  style examples (never the question being asked), and the AI is asked for short, direct answers.
  Word match still compares wording with your expected answers, so read the answers too.
- Their answers are stored only in benchmark results. **Nothing an outside AI says is ever used
  for training**: datasets, learning and Build dataset never read benchmark results. Deer AI's own
  models stay trained only by you.
- Questions go to that company and use your API credit with them; keys stay on your server.
- Outside models can only be compared on chat benchmarks.

## Training your first models

1. **Datasets tab**: upload data.
   - *Text* (for chat): `.txt` files. One line per turn:
     ```
     User: What is a buck?
     Deer: A buck is a male deer.
     ```
   - *Images* (for analysis and creation): one `.zip` with a folder per subject:
     `deer/001.jpg`, `fox/002.jpg`, `oak leaf/003.png`. Folder names become labels.
2. **Models tab**: pick a type, size and dataset, then **Create model**.
3. Set **Steps to train** and press **Start training**. Watch loss and samples live.
   **Stop and save** keeps progress; **Continue training** resumes from the checkpoint.
4. Tick **Published for users** so it shows up in chat.

With the starter data, rough CPU timings: image analysis (tiny, 150 steps) ≈ 10 s;
chat (tiny, 1500 steps) ≈ 2 min; image creation (tiny, 700 steps) ≈ 3 min.

## What each model is

| Type | Architecture | Learns from |
|---|---|---|
| Chat | Byte-level GPT transformer (causal self-attention, GELU MLP, LayerNorm) | text |
| Image analysis | 6-layer convolutional network, global pooling, softmax | labeled image folders |
| Image creation | Conditional variational autoencoder; prompt words pick the subject | labeled image folders |

Image analysis also reports measurements that need no training: size, brightness,
contrast, colorfulness, detail, and a k-means color palette.

## Settings

Under *Advanced settings* when creating a model, pass JSON overrides, e.g.
`{"n_layer": 8, "n_embd": 256, "lr": 0.0005}`. Keys per type:

- chat: `n_layer, n_head, n_embd, block_size, batch_size, lr`
- vision: `image_size, channels, batch_size, lr`
- canvas: `image_size, latent, hidden, batch_size, lr, beta`
- all: `weight_decay, grad_clip, warmup, eval_interval`

Environment variables (or put them in `deer.env`, see `deer.env.example`): `DEER_DATA` (data folder), `DEER_MAX_JOBS` (parallel training
jobs, default 2), `DEER_BUSY_SHARE`, `DEER_MIN_FREE_MB`, `DEER_TRAIN_MODE=thread`, `DEER_LEARN_INTERVAL`, `DEER_MAX_ZIP_MB`, `PAYPAL_MODE`, `PAYPAL_CLIENT_ID`,
`PAYPAL_CLIENT_SECRET`, `PAYPAL_WEBHOOK_ID`, `DEER_DEVICE=gpu` (experimental CuPy backend, untested).

## Tests

```bash
pip install pytest httpx
python -m pytest -q tests
```

`test_engine.py` checks every hand-written gradient against numerical derivatives.
`test_api.py` runs the full flow: setup, uploads, real training in worker processes,
resume, stop, chat, analysis, generation, permissions.
`test_updater.py` covers zip updates, backups, GitHub push/pull (against a local stand-in), webhooks, automatic rollback of a broken update, and the installer.
`test_github.py` covers the Pages build, relative paths, cross-site access rules, the workflows and Codespaces config.
`test_feedback_repair.py` guards the Learning inbox against reused message ids and repairs old broken items.
`test_devapi.py` covers API keys, billing per token and picture, PayPal top-ups (no double credit), limits, streaming and usage reports.
`test_jobs.py` checks the overload rules, two jobs at once, the queue, and waiting while the server is busy.
`test_external.py` runs other-AI comparisons against a local stand-in server (OpenAI and Anthropic formats), answer sheets, and checks nothing leaks into training.
`test_bench.py` covers benchmarks, the multi-line chat format, code/project uploads and stop markers.
`test_users_train.py` checks taught lessons are really recalled, user data collection, opt-out and hidden models.
`test_teach.py` covers starting from a model, teachers, and models as data sources.
`test_curate.py` covers combining, filtering, class renaming/merging and training on the result.
`test_features.py` covers streaming speed, cache headers, test mode, usage and saved items.
`test_learning.py` covers feedback, review, dataset and class growth, auto-learn and rollback.
`test_billing.py` runs the payment flow against a fake PayPal: plan linking, subscribe,
upgrade, usage limits, webhook signatures, renewals, suspension, cancel, price changes.

## Honest limits

Training runs on the CPU, so models stay small. Chat quality depends almost entirely
on how much conversation text you give it; generated images are 32–64 px. The
architecture is the same family used by large systems, so it scales up with more data,
compute, and a GPU backend.

## Layout

```
deer/engine/    tensor + autograd, layers, AdamW
deer/models/    DeerChat, DeerVision, DeerCanvas
deer/training.py  training worker (checkpoints, backups, class growth, guarded fine-tunes)
deer/learning.py  user feedback, review inbox, auto-learn scheduler
deer/curate.py    combine + filter datasets into new ones
deer/bench.py     benchmarks and scoring
deer/external.py  other AI models, for benchmark comparison only
deer/updater.py   updates from the admin panel: zip, GitHub, backups, self-healing restart
deer/apikeys.py   developer API keys, prices, credit and usage (routes in deer/server/devapi.py)
deer/chatfmt.py   the chat text format (multi-line replies, end markers)
deer/inference.py serving + checkpoint hot-reload
deer/server/    FastAPI app, auth, billing routes, training scheduler (jobs.py), SQLite
deer/billing/   plans, usage limits, PayPal client
static/         chat UI and admin UI
tools/          starter data generator, VPS installer, Termux scripts, GitHub Pages build
.github/        GitHub Actions: test + deploy to the VPS, publish to GitHub Pages
.devcontainer/  GitHub Codespaces setup
```
