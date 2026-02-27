# from dotenv import load_dotenv

# from livekit import agents, rtc
# from livekit.agents import AgentServer, AgentSession, Agent, room_io
# from livekit.plugins import (
#     openai,
#     noise_cancellation,
# )

# from livekit.plugins import google
# from prompts import AGENT_INSTRUCTION, SESSION_INSTRUCTION
# from tool import getWeather, searchWeb
# from tool import send_email

# load_dotenv(".env")


# class Assistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions=AGENT_INSTRUCTION,
#             llm=google.beta.realtime.RealtimeModel(voice="Charon", temperature=0.8),
#             tools=[getWeather, searchWeb, send_email],
#         )


# server = AgentServer()


# @server.rtc_session()
# async def my_agent(ctx: agents.JobContext):
#     session = AgentSession()

#     await session.start(
#         room=ctx.room,
#         agent=Assistant(),
#         room_options=room_io.RoomOptions(
#             audio_input=room_io.AudioInputOptions(
#                 noise_cancellation=lambda params: (
#                     noise_cancellation.BVCTelephony()
#                     if params.participant.kind
#                     == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
#                     else noise_cancellation.BVC()
#                 ),
#             ),
#         ),
#     )

#     await session.generate_reply(instructions=SESSION_INSTRUCTION)


# if __name__ == "__main__":
#     agents.cli.run_app(server)


import os
from dotenv import load_dotenv
from google import genai

from prompts import AGENT_INSTRUCTION
from tool import getWeather, searchWeb, send_email

load_dotenv(".env")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


import asyncio
from google import genai
from dotenv import load_dotenv

from prompts import AGENT_INSTRUCTION
from tool import getWeather, searchWeb, send_email

load_dotenv()

client = genai.Client()
MODEL = "models/gemini-flash-latest"


class Assistant:
    def __init__(self):
        self.client = client

    def run(self, user_text: str) -> str:
        """
        1. Ask Gemini what to do (normal reply OR function call)
        2. If function call → execute the async tool
        3. Send tool result back to Gemini
        4. Return final natural-language response
        """

        # ---------- STEP 1: INITIAL MODEL CALL ----------
        response = self.client.models.generate_content(
            model=MODEL,
            contents=[
                {"role": "user", "parts": [AGENT_INSTRUCTION]},
                {"role": "user", "parts": [user_text]},
            ],
        )

        part = response.candidates[0].content.parts[0]

        # ---------- STEP 2: TOOL CALL ----------
        if hasattr(part, "function_call"):
            tool_name = part.function_call.name
            tool_args = part.function_call.args

            tool_result = self._execute_tool(tool_name, tool_args)

            # ---------- STEP 3: FEED RESULT BACK ----------
            followup = self.client.models.generate_content(
                model=MODEL,
                contents=[
                    {"role": "user", "parts": [AGENT_INSTRUCTION]},
                    {
                        "role": "user",
                        "parts": [
                            f"The user asked: {user_text}\n"
                            f"The tool returned this result:\n{tool_result}\n\n"
                            "Respond clearly and naturally to the user."
                        ],
                    },
                ],
            )

            return followup.text

        # ---------- STEP 4: NORMAL RESPONSE ----------
        return response.text

    def _execute_tool(self, name: str, args: dict) -> str:
        """
        Executes async tools safely from sync code.
        """

        if name == "getWeather":
            return asyncio.run(getWeather(None, **args))

        if name == "searchWeb":
            return asyncio.run(searchWeb(None, **args))

        if name == "send_email":
            return asyncio.run(send_email(None, **args))

        return "Requested tool is not available."
