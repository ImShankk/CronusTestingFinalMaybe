import logging
from typing import Optional
from livekit.agents import function_tool, RunContext
import requests
from langchain_community.tools import DuckDuckGoSearchRun
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


@function_tool()
async def getWeather(context: RunContext, city: str) -> str:
    """
    Get current weather using Open-Meteo (stable, no API key).
    """
    try:
        # Step 1: geocode city → latitude & longitude
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo = requests.get(geo_url, timeout=5).json()

        if "results" not in geo or not geo["results"]:
            return f"I couldn’t find the location '{city}'."

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        name = geo["results"][0]["name"]

        # Step 2: current weather
        weather_url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current_weather=true"
        )

        weather = requests.get(weather_url, timeout=5).json()
        cw = weather.get("current_weather", {})

        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        code = cw.get("weathercode")

        # Round values
        temp_rounded = round(temp) if temp is not None else temp
        wind_rounded = round(wind) if wind is not None else wind

        return (
            f"Current weather in {name}: {temp_rounded}°C, "
            f"wind {wind_rounded} km/h. "
            f"Weather code {code}."
        )

    except Exception as e:
        logging.error(f"Open-Meteo error: {e}")
        return f"Sorry, I couldn't fetch weather for {city}."


# @function_tool()
# async def searchWeb(context: RunContext, query: str) -> str:
#     """
#     Search the web using DuckDuckGo and return the top results.
#     """
#     try:
#         results = DuckDuckGoSearchRun().run(tool_input=query)
#         logging.info(f"Web search completed successfully for {query}: {results}")
#         return results
#     except Exception as e:
#         logging.error(f"An error occurred during web search for {query}: {e}")
#         return f"Sorry, An error occured while I was searching the web for '{query}'."


from ddgs import DDGS


@function_tool()
async def searchWeb(context: RunContext, query: str) -> str:
    """
    Perform a web search using the native DDGS library.
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            formatted = "\n".join([f"- {r['title']}: {r['href']}" for r in results])

        return formatted or "No results found."
    except Exception as e:
        logging.error(f"DDGS search error for {query}: {e}")
        return f"Sorry, I couldn't search the web for '{query}'."


@function_tool()
async def send_email(
    context: RunContext,
    to_email: str,
    subject: str,
    message: str,
    cc_email: Optional[str] = None,
) -> str:
    """
    Send an email through Gmail.

    Args:
        to_email: Recipient email address
        subject: Email subject line
        message: Email body content
        cc_email: Optional CC email address
    """
    try:
        # Gmail SMTP configuration
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        # Get credentials from environment variables
        gmail_user = os.getenv("GMAIL_USER")
        gmail_password = os.getenv(
            "GMAIL_APP_PASSWORD"
        )  # Use App Password, not regular password

        if not gmail_user or not gmail_password:
            logging.error("Gmail credentials not found in environment variables")
            return "Email sending failed: Gmail credentials not configured."

        # Create message
        msg = MIMEMultipart()
        msg["From"] = gmail_user
        msg["To"] = to_email
        msg["Subject"] = subject

        # Add CC if provided
        recipients = [to_email]
        if cc_email:
            msg["Cc"] = cc_email
            recipients.append(cc_email)

        # Attach message body
        msg.attach(MIMEText(message, "plain"))

        # Connect to Gmail SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable TLS encryption
        server.login(gmail_user, gmail_password)

        # Send email
        text = msg.as_string()
        server.sendmail(gmail_user, recipients, text)
        server.quit()

        logging.info(f"Email sent successfully to {to_email}")
        return f"Email sent successfully to {to_email}"

    except smtplib.SMTPAuthenticationError:
        logging.error("Gmail authentication failed")
        return "Email sending failed: Authentication error. Please check your Gmail credentials."
    except smtplib.SMTPException as e:
        logging.error(f"SMTP error occurred: {e}")
        return f"Email sending failed: SMTP error - {str(e)}"
    except Exception as e:
        logging.error(f"Error sending email: {e}")
        return f"An error occurred while sending email: {str(e)}"
