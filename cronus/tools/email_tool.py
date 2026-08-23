"""Email over SMTP.

Credentials are read from configuration inside the handler and never appear in
a prompt, a tool schema, a tool result, or a log line. Sending is a CONFIRM
tool, so the runtime always shows the draft and waits for approval first.
"""

from __future__ import annotations

import hashlib
import re
import smtplib
from email.message import EmailMessage
from email.utils import formataddr, parseaddr
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger
from .base import RiskLevel, Tool, ToolContext, ToolResult, object_schema

log = get_logger("tools.email")

_ADDRESS_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")
_MAX_ATTACHMENT_BYTES = 5_000_000
_MAX_SUBJECT = 200
# Key under which this session records what it has already sent.
_SENT_KEY = "sent_emails"


def _clean_header(value: str) -> str | None:
    """Reject header text carrying newlines.

    A subject can originate in text the model read from a web page, so a
    smuggled CR/LF is a header-injection attempt. Python's email library
    raises on these; catching it here turns a stack trace into a refusal.
    """
    text = (value or "").strip()
    if any(char in text for char in ("\r", "\n", "\x00")):
        return None
    return text[:_MAX_SUBJECT]


def _fingerprint(to: str, subject: str, body: str) -> str:
    joined = "\x00".join((to, subject, body))
    return hashlib.sha256(joined.encode('utf-8')).hexdigest()


# What the user might call their own mailbox. Cronus sends from the user's
# account, so "yourself" resolves there too.
_SELF_ALIASES = frozenset(
    {
        "me", "myself", "self", "yourself", "my email", "my email address",
        "my inbox", "my address", "my account",
    }
)


def _valid_address(raw: str) -> str | None:
    _, address = parseaddr(raw or "")
    return address if address and _ADDRESS_RE.match(address) else None


def _resolve_recipient(raw: str, account: str | None) -> str | None:
    """Turn a recipient into a real address, expanding self-references.

    Without this the model has to guess what "me" means, and a guess is how a
    message ends up addressed to an invented domain.
    """
    candidate = (raw or "").strip().strip(".").lower()
    if candidate in _SELF_ALIASES:
        return _valid_address(account or "")
    return _valid_address(raw)


def send_email(
    to: str,
    subject: str,
    body: str,
    cc: str | None = None,
    attachment_path: str | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Send a message. The runtime has already confirmed it with the user."""
    if context is None:
        return ToolResult.failure("Email is not available in this context.")

    settings = context.config.email
    if not settings.configured:
        return ToolResult.failure(
            "Email is not set up. Tell the user to set GMAIL_USER and "
            "GMAIL_APP_PASSWORD in their .env file."
        )

    recipient = _resolve_recipient(to, settings.user)
    if recipient is None:
        return ToolResult.failure(f"{to!r} is not a valid email address.")

    recipients = [recipient]
    cc_address = None
    if cc:
        cc_address = _resolve_recipient(cc, settings.user)
        if cc_address is None:
            return ToolResult.failure(f"The CC address {cc!r} is not valid.")
        recipients.append(cc_address)

    clean_subject = _clean_header(subject)
    if clean_subject is None:
        return ToolResult.failure(
            "The subject line contains a line break, which is not allowed in an "
            "email header. Rewrite it as a single line."
        )

    # The model can decide a send failed when it did not, and try again. An
    # identical message already sent in this session is refused rather than
    # delivered twice.
    already_sent = context.session.setdefault(_SENT_KEY, set())
    fingerprint = _fingerprint(recipient, clean_subject, body or "")
    if fingerprint in already_sent:
        log.warning("refused duplicate send to=%s", recipient)
        return ToolResult.failure(
            f"That exact email was already sent to {recipient} in this session. "
            "It was not sent again. Tell the user it has already gone out."
        )

    message = EmailMessage()
    message["From"] = formataddr(("", settings.user or ""))
    message["To"] = recipient
    if cc_address:
        message["Cc"] = cc_address
    message["Subject"] = clean_subject or "(no subject)"
    message.set_content(body or "")

    if attachment_path:
        attached = _attach(message, attachment_path, context)
        if attached is not None:
            return attached

    context.progress(f"Sending email to {recipient}")
    try:
        with smtplib.SMTP(
            settings.smtp_host, settings.smtp_port, timeout=settings.timeout
        ) as server:
            server.starttls()
            server.login(settings.user, settings.app_password)
            server.send_message(message, to_addrs=recipients)
    except smtplib.SMTPAuthenticationError:
        log.error("smtp authentication rejected for configured account")
        return ToolResult.failure(
            "The mail server rejected the login. The app password in .env is "
            "probably wrong or expired."
        )
    except smtplib.SMTPRecipientsRefused:
        return ToolResult.failure(f"The mail server refused the address {recipient}.")
    except (smtplib.SMTPException, OSError) as exc:
        log.error("smtp send failed: %s", type(exc).__name__)
        return ToolResult.failure(f"The email could not be sent: {type(exc).__name__}.")

    already_sent.add(fingerprint)
    log.info("email sent to=%s subject=%r", recipient, clean_subject)
    return ToolResult(
        content=f"Email sent to {recipient} with subject {clean_subject!r}.",
        display=f"sent to {recipient}",
        data={"to": recipient, "subject": clean_subject},
    )


def _attach(
    message: EmailMessage, raw_path: str, context: ToolContext
) -> ToolResult | None:
    """Attach a file, or return the failure that stops the send."""
    if context.paths is None:
        return ToolResult.failure("File access is off, so I can't attach anything.")
    try:
        path: Path = context.paths.resolve(raw_path, must_exist=True)
    except Exception as exc:
        return ToolResult.failure(str(getattr(exc, "user_message", exc)))

    size = path.stat().st_size
    if size > _MAX_ATTACHMENT_BYTES:
        return ToolResult.failure(
            f"{path.name} is {size // 1_000_000} MB, which is too large to attach."
        )
    message.add_attachment(
        path.read_bytes(),
        maintype="application",
        subtype="octet-stream",
        filename=path.name,
    )
    return None


def draft_email(
    to: str, subject: str, body: str, context: ToolContext | None = None
) -> ToolResult:
    """Compose a message for review. Nothing is stored or sent.

    This is a preview shown in the conversation, not a draft saved in the
    user's mail account.
    """
    account = context.config.email.user if context is not None else None
    recipient = _resolve_recipient(to, account)
    if recipient is None:
        if (to or "").strip().lower() in _SELF_ALIASES:
            return ToolResult.failure(
                "I don't know the user's own email address because no account is "
                "configured. Ask them for the address to use."
            )
        return ToolResult.failure(
            f"{to!r} is not a valid email address. Ask the user for the real one "
            "rather than guessing."
        )

    warning = ""
    if context is not None and not context.config.email.configured:
        # Saying "draft ready" while sending is impossible is misleading.
        warning = (
            "\n\nNote: no email account is configured, so this cannot actually be "
            "sent yet. Tell the user to set GMAIL_USER and GMAIL_APP_PASSWORD."
        )

    return ToolResult(
        content=(
            f"Draft ready.\nTo: {recipient}\nSubject: {subject}\n\n{body}\n\n"
            "Show this to the user and ask whether to send it. Call send_email "
            f"only once they say yes.{warning}"
        ),
        display=f"drafted to {recipient}",
        data={"to": recipient, "subject": subject, "body": body},
    )


def _preview(arguments: dict[str, Any]) -> str:
    # Just the question. Who it goes to, the subject, and the whole body are
    # shown underneath it as the message itself, which is what a person needs
    # to read before saying yes.
    return "Send this email?"


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="draft_email",
            description=(
                "Write an email draft without sending it. Use this when the user "
                "may want to review or change the wording first."
            ),
            parameters=object_schema(
                {
                    "to": {
                        "type": "string",
                        "description": (
                            "Recipient email address, or 'me' for the user's own "
                            "account. Never invent an address."
                        ),
                    },
                    "subject": {"type": "string", "description": "Subject line."},
                    "body": {"type": "string", "description": "Message body as plain text."},
                },
                required=["to", "subject", "body"],
            ),
            handler=draft_email,
            risk=RiskLevel.SAFE,
            category="communication",
        ),
        Tool(
            name="send_email",
            description=(
                "Send an email from the user's configured account. The user is "
                "always shown the draft and asked to approve before it goes out."
            ),
            parameters=object_schema(
                {
                    "to": {
                        "type": "string",
                        "description": (
                            "Recipient email address, or 'me' for the user's own "
                            "account. Never invent an address."
                        ),
                    },
                    "subject": {"type": "string", "description": "Subject line."},
                    "body": {"type": "string", "description": "Message body as plain text."},
                    "cc": {"type": "string", "description": "Optional CC address."},
                    "attachment_path": {
                        "type": "string",
                        "description": "Optional path to a file to attach.",
                    },
                },
                required=["to", "subject", "body"],
            ),
            handler=send_email,
            risk=RiskLevel.CONFIRM,
            category="communication",
            timeout=40.0,
            preview=_preview,
        ),
    ]
