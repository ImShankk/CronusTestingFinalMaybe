"""Email over SMTP.

Credentials are read from configuration inside the handler and never appear in
a prompt, a tool schema, a tool result, or a log line. Sending is a CONFIRM
tool, so the runtime always shows the draft and waits for approval first.
"""

from __future__ import annotations

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


def _valid_address(raw: str) -> str | None:
    _, address = parseaddr(raw or "")
    return address if address and _ADDRESS_RE.match(address) else None


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

    recipient = _valid_address(to)
    if recipient is None:
        return ToolResult.failure(f"{to!r} is not a valid email address.")

    recipients = [recipient]
    cc_address = None
    if cc:
        cc_address = _valid_address(cc)
        if cc_address is None:
            return ToolResult.failure(f"The CC address {cc!r} is not valid.")
        recipients.append(cc_address)

    message = EmailMessage()
    message["From"] = formataddr(("", settings.user or ""))
    message["To"] = recipient
    if cc_address:
        message["Cc"] = cc_address
    message["Subject"] = subject or "(no subject)"
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

    log.info("email sent to=%s subject=%r", recipient, subject)
    return ToolResult(
        content=f"Email sent to {recipient} with subject {subject!r}.",
        display=f"sent to {recipient}",
        data={"to": recipient, "subject": subject},
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


def draft_email(to: str, subject: str, body: str, context: ToolContext | None = None) -> ToolResult:
    """Hold a draft in session state so the user can revise it before sending."""
    draft: dict[str, Any] = {"to": to, "subject": subject, "body": body}
    if context is not None:
        context.session["email_draft"] = draft
    return ToolResult(
        content=(
            f"Draft ready.\nTo: {to}\nSubject: {subject}\n\n{body}\n\n"
            "Show this to the user and ask whether to send it. Call send_email "
            "only once they say yes."
        ),
        display=f"drafted to {to}",
        data=draft,
    )


def _preview(arguments: dict[str, Any]) -> str:
    recipient = arguments.get("to", "(nobody)")
    subject = arguments.get("subject", "(no subject)")
    return f"Send an email to {recipient} with the subject {subject!r}?"


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
                    "to": {"type": "string", "description": "Recipient email address."},
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
                    "to": {"type": "string", "description": "Recipient email address."},
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
