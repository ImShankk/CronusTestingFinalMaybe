# Cronus

A personal AI assistant that runs on your own machine. Cronus understands what
you want, decides which tools to use, chains them together across several
steps, remembers what matters, asks before doing anything consequential, and
talks if you want it to.

```
you  What's the weather in Edmonton tomorrow?
     Using get_weather...
cronus  Tomorrow in Edmonton: overcast, high of 26, low of 16.

you  Remind me to bring an umbrella at 8.
cronus  Done. I'll remind you tomorrow at 08:00.

you  Email john@example.com that I'll be 15 minutes late.

     ┌ Confirm ─────────────────────────────────────────┐
     │ Send an email to john@example.com with the       │
     │ subject 'Running late'?                          │
     │ to: john@example.com                             │
     │ body: Hi John, I'm running about 15 minutes      │
     │ late. See you shortly.                           │
     └──────────────────────────────────────────────────┘
     Go ahead? [y/n]
```

---

## Architecture

Cronus is a core with interfaces attached to it, not a script. The CLI is one
interface; a GUI or an HTTP API can be another without touching anything below
it.

```
      CLI  (/ future GUI, API)
       │      events, confirmation prompts
       ▼
  Assistant runtime  ──►  Context builder  ──►  LLM provider ──► Gemini
       │                    profile,                 ▲
       │                    memories,                │  tool schemas,
       │                    history                  │  tool results
       ▼                                             │
  Permission policy ──► Confirmation ──► Tool registry
                                              │
             ┌────────────┬────────────┬──────┴─────┬───────────┐
           web        weather       email        files      system
                                              memory     reminders
                                                  │
                                             SQLite store
```

One turn of the agent loop:

```
user input → context → model → tools requested?
                                  │
                     no ──────────┴────────── yes
                     │                          │
                   reply            permission check (in code)
                                                │
                                     confirm if consequential
                                                │
                                          execute tool
                                                │
                                    result back to the model ──┐
                                                               │
                                          (repeat, bounded) ◄──┘
```

| Module | What lives there |
| --- | --- |
| `cronus/core/` | agent runtime, conversation state, context assembly, events, prompts |
| `cronus/llm/` | provider-neutral message types and the Gemini implementation |
| `cronus/tools/` | tool definitions, registry, schema validation, and the tools themselves |
| `cronus/memory/` | long-term memory with relevance-ranked recall |
| `cronus/security/` | permission policy, confirmation, filesystem containment |
| `cronus/voice/` | speech-to-text, text-to-speech, wake word |
| `cronus/automation/` | reminders and recurring tasks |
| `cronus/storage/` | SQLite schema and connections |
| `cronus/interfaces/` | the CLI |

---

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/ImShankk/CronusTestingFinalMaybe
cd CronusTestingFinalMaybe

python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

copy .env.example .env         # macOS/Linux: cp .env.example .env
```

Put a Gemini API key in `.env` — get one free at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey):

```
GOOGLE_API_KEY=your-key-here
```

That is the only required setting.

---

## Run

```bash
python main.py                 # type to Cronus
python main.py --speak         # type, hear replies
python main.py --voice         # speak and listen
python main.py -m "what's the weather in Oslo?"   # one message, then exit
```

In-session commands:

| Command | Does |
| --- | --- |
| `/tools` | list every tool and its permission level |
| `/memory` | show what Cronus remembers |
| `/forget <id>` | delete one memory |
| `/profile` | show the stored user profile |
| `/tasks` | list scheduled reminders |
| `/voice` | toggle spoken replies |
| `/clear` | start a fresh conversation |
| `/quit` | exit |

---

## Tools

| Tool | Permission | What it does |
| --- | --- | --- |
| `search_web` | allow | search the web, with source URLs kept |
| `read_webpage` | allow | fetch one page and read its text |
| `get_weather` | allow | current conditions and up to 7 days of forecast |
| `system_status` | allow | battery, memory, disk, CPU, uptime |
| `open_url` | allow | open a link in the browser |
| `open_app` | allow | launch an allowlisted application |
| `draft_email` | allow | write an email without sending it |
| `send_email` | **confirm** | send it, after showing you the draft |
| `list_directory` `read_file` `search_files` | allow | read inside allowed folders |
| `write_file` `move_file` `delete_file` | **confirm** | change files, after asking |
| `remember_this` `set_preference` | allow | store a fact or a profile setting |
| `recall_memories` `list_memories` | allow | look up what is stored |
| `forget_memory` | **confirm** | delete a memory |
| `create_reminder` `list_reminders` | allow | schedule and review reminders |
| `cancel_reminder` | **confirm** | cancel one |

### Adding a tool

Write a function, describe it, and add it to a module's `build_tools()`. The
registry handles schemas, validation, timeouts, and errors; the runtime handles
permissions and confirmation. Nothing else changes.

```python
# cronus/tools/mytools.py
from .base import RiskLevel, Tool, ToolResult, object_schema

def flip_a_coin(sides: int = 2) -> ToolResult:
    import random
    return ToolResult(content=f"It came up {random.randint(1, sides)}.")

def build_tools() -> list[Tool]:
    return [
        Tool(
            name="flip_a_coin",
            description="Pick a random number between 1 and sides.",
            parameters=object_schema(
                {"sides": {"type": "integer", "minimum": 2, "default": 2}}
            ),
            handler=flip_a_coin,
            risk=RiskLevel.SAFE,
            category="fun",
        )
    ]
```

Then add the module to `TOOL_MODULES` in `cronus/app.py`. A handler that
declares a `context` parameter receives a `ToolContext` with configuration,
memory, the scheduler, and the path guard.

---

## Permissions and confirmation

**The model is not a security boundary.** Every tool declares a risk level in
code, and the permission policy — not the prompt — decides what happens.

| Risk | Default | Examples |
| --- | --- | --- |
| `SAFE` | run | weather, search, open a link |
| `LOW` | run | read a file, store a memory, set a reminder |
| `CONFIRM` | ask first | send email, write/move/delete a file |
| `HIGH` | refuse unless opted into | reserved for anything genuinely dangerous |
| `BLOCKED` | never runs | cannot be re-enabled by configuration |

Override a tool in `.env`:

```
CRONUS_TOOL_PERMISSIONS=send_email=allow,delete_file=block
```

A confirmation is a real object with state and an expiry, not a prompt
instruction. It shows the actual arguments — the whole email body, the exact
path — so you approve what will really happen. Declining reports back to the
model, which acknowledges it rather than retrying.

### Filesystem safety

File tools only work inside allowlisted roots (`CRONUS_FILE_ROOTS`, defaulting
to Documents, Desktop, and Downloads). Every path is fully resolved before use,
so `../..`, symlinks, and absolute paths elsewhere are all rejected. Reads are
capped in size and limited to known text extensions.

### Untrusted content

Web pages, search results, and file contents are labelled as data where they
enter the conversation, and the system instruction tells the model to describe
instructions found in them rather than follow them. Cronus has **no shell and no
arbitrary code execution** — `open_app` launches only allowlisted programs.

---

## Memory

Memory is selective. Nothing is stored unless the model explicitly proposes it
and the application accepts it; conversations are never dumped in wholesale.
Near-duplicates merge instead of piling up, and the store is capped.

```
you  Remember that I prefer concise answers.
cronus  Got it.

you  Explain photosynthesis.        ← the preference is applied without asking
```

Recall uses SQLite's full-text index with keyword scoring — no embedding model,
works offline. Preferences always surface, since they shape every answer.
Everything is inspectable and deletable through `/memory` and `/forget`.

### Context

Each request is assembled from separate layers — instructions, profile,
relevant memories, running summary, recent turns — inside a character budget
(`CRONUS_CONTEXT_BUDGET`). Turns that fall outside the window are folded into a
one-off summary rather than replayed, so a long session does not inflate every
request.

---

## Voice

```bash
python main.py --voice
```

* **Speech-to-text** — `SpeechRecognition` with Google's free web endpoint.
* **Speech-to-speech** — [Piper](https://github.com/rhasspy/piper/releases),
  local and offline. Point `CRONUS_PIPER_EXE` and `CRONUS_PIPER_MODEL` at the
  binary and a voice model. Without Piper, Windows SAPI is used automatically.
* Replies are cleaned before speaking: no markdown, no URLs read out character
  by character.
* Speech is interruptible — Ctrl+C cuts it off mid-sentence.

### Wake word

```
CRONUS_WAKE_WORD_ENABLED=true
CRONUS_WAKE_WORD=hey cronus
```

**How it actually works, and its limitation.** Cronus listens in short bursts
and activates when a transcript starts with the wake phrase, matched fuzzily so
"hey chronos" still works. It carries the rest of the sentence through, so "hey
cronus, what's the weather" works in one breath.

This is *not* always-on, low-power keyword spotting like Porcupine or
openWakeWord. It sends short clips to the same speech service used for
dictation, so it needs a network connection and costs a round trip per burst.
It is real and it works; it is not free. `WakeWordDetector` is an interface — a
dedicated detector can be dropped in without touching anything else. When
hands-free listening is off or unavailable, Cronus falls back to press-Enter
push-to-talk.

---

## Reminders

```
you  Remind me to check my applications every Friday.
cronus  Set for every Friday at 09:00.
```

Tasks live in SQLite and survive restarts. A background thread fires them and
the CLI announces them (and speaks them, if speech is on). Recurrence
supports `daily`, `hourly`, `weekly:monday,friday`, and `every:30m` / `every:2h`.

---

## Configuration

Everything is environment-driven; see `.env.example` for the annotated list.
The essentials:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` | — | **required** |
| `CRONUS_MODEL` | `gemini-flash-latest` | which Gemini model |
| `CRONUS_DATA_DIR` | `~/.cronus` | database and logs |
| `CRONUS_FILE_ROOTS` | Documents, Desktop, Downloads | folders file tools may use |
| `CRONUS_TOOL_PERMISSIONS` | — | per-tool permission overrides |
| `CRONUS_MAX_TOOL_ITERATIONS` | `8` | tool round trips per request |
| `CRONUS_VOICE` | `false` | speech on by default |
| `CRONUS_LOG_LEVEL` | `INFO` | developer log verbosity |
| `GMAIL_USER` / `GMAIL_APP_PASSWORD` | — | email ([app password](https://myaccount.google.com/apppasswords), not your account password) |

Missing or malformed configuration fails with a sentence telling you what to
fix, not a traceback.

### Logging

Developer logs go to `~/.cronus/logs/cronus.log` (rotating). The terminal shows
only what you need. A redaction filter scrubs API keys and passwords from log
records as a backstop.

---

## Testing

```bash
python -m pytest
```

225 tests, about three seconds, **no API key, no network, and no microphone
required** — the model is a scripted fake, external services are mocked, and
voice providers are tested through their interfaces.

Covered: the agent loop (chaining, parallel calls, tool failure recovery,
iteration limits, cancellation), the tool registry (validation, timeouts, async
handlers), permissions and confirmation, path containment and traversal,
memory CRUD and persistence, context budgeting and summarisation, the scheduler,
the Gemini translation layer, configuration errors, and realistic end-to-end
conversations.

---

## Known limitations

* **Free-tier Gemini quotas are small.** The default model allows a handful of
  requests per minute and per day. Cronus honours the server's retry hint on a
  rate limit and fails fast with a clear message on a daily quota. Set
  `CRONUS_MODEL` to another model if you hit it.
* **The wake word is not always-on keyword spotting.** See above.
* **Speech-to-text needs a network connection** — the free Google endpoint is
  not local. Piper speech output *is* local.
* **Email is send-only.** Cronus can draft, send, and attach a file; it cannot
  read your inbox or reply to a thread.
* **A timed-out tool is abandoned, not killed.** Python cannot safely kill a
  thread, so handlers set their own network timeouts as well.
* **No calendar integration.** Reminders are Cronus's own, not your calendar's.

## Roadmap

* Read email, so "reply to John" works end to end
* An HTTP interface, so the core can back a desktop or web UI
* A local wake-word detector, for genuinely always-on activation
* Local speech-to-text via `faster-whisper`, removing the network dependency
* Richer plans for long multi-step tasks, with visible progress
