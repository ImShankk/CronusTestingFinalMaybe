# Cronus

A personal AI assistant that runs on your own machine. Cronus understands what
you want, decides which tools to use, chains them together across several
steps, remembers what matters, asks before doing anything consequential, and
talks if you want it to.

```
you  What's the weather tomorrow?
     Checking the weather...
cronus  Tomorrow in Edmonton: light showers, high of 22, low of 15.

you  What about Saturday?
cronus  Cooler and damp -- 16 and some light drizzle.

you  Email john@example.com that I'll be 15 minutes late.

     ┌ Confirm ─────────────────────────────────────────┐
     │ Send this email?                                 │
     │ To: john@example.com                             │
     │ Subject: Running late                            │
     │                                                  │
     │ Hi John, I'm running about 15 minutes late.      │
     │ See you shortly.                                 │
     └──────────────────────────────────────────────────┘
     Go ahead? [y/n] yes, go ahead
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
| `/clear` | forget the current conversation, here and on disk |
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
memory, the user profile, the scheduler, the path guard, and per-session
scratch space.

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

Consequential actions also guard against being done twice. An identical email
already sent in this session is refused rather than delivered again, and asking
twice for the same reminder updates the existing one instead of creating a
second — a model that wrongly believes a step failed cannot repeat it.

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

## Location and time

Cronus never guesses where you are. `CRONUS_LOCATION` is put into its context
and used by the weather tool when you don't name a city, and `CRONUS_TIMEZONE`
drives the clock, "tomorrow", and reminder times:

```
CRONUS_TIMEZONE=America/Edmonton
CRONUS_LOCATION=Edmonton, Alberta, Canada
```

With neither set, Cronus is told its location is unknown and asks rather than
picking a city. You can also just tell it — "I'm in Edmonton" — and it stores
that in your profile, which takes precedence over the configured value.

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

Recall is widened by the last couple of turns, not just the sentence you typed.
"What do you call me?" carries almost no searchable words of its own; the
exchange it sits in does. A memory can only be helped into range by that, never
pushed out of it.

### Continuity

Closing the terminal is not the same as being forgotten. The running summary
and the last six turns are kept in `conversation_state` and restored on the
next launch, so "what were we talking about?" still works tomorrow. Cronus is
told how long ago that was, so it picks up rather than pretending no time
passed.

This is continuity, not an archive: only the tail is kept, `/clear` deletes it,
and none of it becomes long-term memory — that is still written only when the
model explicitly asks to store something.

---

## Voice

```bash
python main.py --voice          # or set CRONUS_VOICE=true
```

### Modes

`CRONUS_VOICE_MODE` chooses how a turn begins. All three share one runtime;
they differ only in that first step.

| Mode | How a turn starts |
| --- | --- |
| `continuous` (default) | Cronus listens again after every reply. No keypress. |
| `push_to_talk` | Press Enter before each turn. Useful when debugging, or in a noisy room. |
| `wake_word` | Stays idle until it hears "hey cronus". |

The header tells you which is active:

```
gemini-flash-latest · 22 tools · voice: continuous · interruptible · /help
Listening...
```

### Interruption (barge-in)

Talking over Cronus stops it mid-sentence and starts listening. This is real
cancellation, not a delayed reaction: speech plays on a worker thread while
the microphone is watched, and the provider's own `stop()` cuts playback.

What you said is then kept and answered. Interrupting is a request, not just a
stop button, so the microphone reopens the instant speech stops and the
utterance is carried into the next turn — you do not repeat yourself. The
monitor only measures loudness, so the syllable or two that triggered the
interrupt is genuinely lost; everything after it is not.

The trigger adapts to the room. Rather than scaling a one-off calibration,
Cronus measures the noise floor live during the first 0.4s of each reply --
which already includes its own voice coming back through your speakers -- and
requires you to be `CRONUS_BARGE_IN_SENSITIVITY` times louder than that, for
at least `CRONUS_MIN_SPEECH_SECONDS`, tolerating the short dips between
syllables. Set `CRONUS_BARGE_IN=false` to switch it off.

Headphones make this far more reliable than speakers: with speakers the
microphone hears Cronus, so the bar it sets for you is higher.

### Not being cut off

Endpointing is tuned for natural speech rather than dictation.
`CRONUS_PAUSE_THRESHOLD` (default 1.0s) is how much silence ends a phrase --
raise it if Cronus interrupts you when you pause to think.
`CRONUS_PHRASE_TIME_LIMIT` (default 30s) is the longest single utterance.

* **Speech-to-text** — `SpeechRecognition` with Google's free web endpoint.
* **Text-to-speech** — [Piper](https://github.com/rhasspy/piper/releases),
  local and offline. `CRONUS_PIPER_EXE` is the executable itself;
  `CRONUS_PIPER_MODEL` is a `.onnx` voice with its `.onnx.json` beside it
  (Piper finds the JSON itself — never point at the `.json`). Voices come from
  [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices); a
  sensible layout is `<install>/piper/piper.exe` and `<install>/voices/`.
  Without Piper, Windows SAPI is used and the header says so.

  **Choosing a voice.** `high` voices sound better but synthesise roughly four
  times slower, and Cronus starts `piper.exe` once per reply, so that cost is
  paid every turn. On a mid-range CPU a `high` voice adds about a second to a
  short reply and several seconds to a long one; `medium` is close to instant.
  Try both — swapping is one line of `.env`.

  **Speaking speed.** `CRONUS_SPEECH_RATE` maps to Piper's `--length-scale`
  (higher rate, shorter audio). 1.25 is a good conversational pace; the effect
  flattens out beyond roughly 2.0.
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
| `CRONUS_TIMEZONE` | system | IANA name, e.g. `America/Edmonton` |
| `CRONUS_LOCATION` | — | where you are, for weather and local questions |
| `CRONUS_VOICE_MODE` | `continuous` | `continuous`, `push_to_talk`, or `wake_word` |
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

419 tests, about nine seconds, **no API key, no network, and no microphone
required** — the model is a scripted fake, external services are mocked, and
voice providers are tested through their interfaces.

Covered: the agent loop (chaining, parallel calls, tool failure recovery,
iteration limits, cancellation, malformed and hostile model output), the tool
registry (validation, timeouts, async handlers), permissions and confirmation,
path containment and traversal, unsafe URL schemes, email header injection and
duplicate sends, memory relevance and duplicate suppression, context budgeting
and summarisation, the scheduler, concurrency across threads, the Gemini
translation layer, configuration errors, and realistic end-to-end
conversations.

`tests/test_hardening.py` holds the regression tests from the post-build audit;
each one pins a defect that was found and fixed. `tests/test_experience.py`
pins the things a user actually perceives: that a follow-up resolves without
repeating yourself, that talking over a reply keeps what you said, that a
session survives a restart, and that tool names never reach the conversation.

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
* **Barge-in depends on your audio setup.** Through speakers the microphone
  also hears Cronus, so the interruption threshold rises and you may need to
  speak up. Headphones avoid this entirely. There is no acoustic echo
  cancellation.
* **Continuous mode hears the room.** Background speech reaches the
  recogniser and costs a transcription round trip, though it rarely produces a
  usable request. Raise `CRONUS_MIC_ENERGY_THRESHOLD`, or use `push_to_talk`
  or `wake_word`, where that matters.

## Roadmap

* Read email, so "reply to John" works end to end
* An HTTP interface, so the core can back a desktop or web UI
* A local wake-word detector, for genuinely always-on activation
* Local speech-to-text via `faster-whisper`, removing the network dependency
* Richer plans for long multi-step tasks, with visible progress
