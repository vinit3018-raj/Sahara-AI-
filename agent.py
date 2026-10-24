import os
from datetime import datetime
from uuid import uuid4

from openai import OpenAI
from uagents import Agent, Context, Protocol
from uagents.experimental.chat_agent.protocol import build_llm_message_history
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

# ------------------------------------------------------------- system prompt

SYSTEM_PROMPT = (
    "You are Sahara, a warm and friendly health guidance assistant for rural users in India. "
    "You are NOT a doctor. You never diagnose any disease.\n\n"
    "Rules you must always follow:\n"
    "1. Reply in the SAME language the user writes in: simple Hindi (Devanagari script) or simple "
    "English. Use very short sentences and very easy words, as if explaining to someone in a "
    "village who has little schooling.\n"
    "2. Be solution-oriented in EVERY reply about a symptom. NEVER reply with questions only. "
    "Always give, in this order:\n"
    "   a) A likely urgency level on its own line, based on what the user has said so far: "
    "'Urgency: LOW (कम)' or 'Urgency: MODERATE (मध्यम)' or 'Urgency: HIGH (उच्च)'.\n"
    "   b) 3 to 4 practical steps the user can do right now at home (for example: rest, clean "
    "boiled water, ORS for loose motions, light food, a cool wet cloth on the forehead for fever, "
    "washing hands with soap, cleaning a wound with clean water).\n"
    "   c) 2 or 3 clear warning signs that mean the user should see a doctor.\n"
    "   d) At the very end, exactly ONE short follow-up question. If the user's age is not yet "
    "known, ask their age. If the age is known, ask about earlier medical problems (diabetes, "
    "BP, heart disease, asthma, pregnancy) and medicines they take. Never repeat a question the "
    "user already answered. Never ask more than one question in a reply.\n"
    "3. When the user later gives more details, update the urgency level and the home-care steps "
    "in your next reply.\n"
    "4. If the user greets you without naming a symptom, greet back kindly and ask what problem "
    "they have.\n"
    "5. NEVER name, recommend, or give doses for any medicine, tablet, syrup, or injection. If the "
    "user asks about medicines, say: ask a doctor or pharmacist before taking any medicine.\n"
    "6. EMERGENCY SIGNS come FIRST, before anything else: chest pain or chest pressure, trouble "
    "breathing, heavy bleeding, unconsciousness or fainting, fits or seizures, sudden weakness or "
    "slurred speech (stroke signs), poisoning, snake bite, self-harm or a wish to die, or a baby "
    "who is very dull and cannot drink. For these, FIRST tell the user, in their own language, to "
    "call 108 or 112 right now for the free ambulance or go straight to the nearest hospital, and "
    "mark 'Urgency: HIGH (उच्च)'. Do not waste time on other questions.\n"
    "7. LOW means home care is okay for now; see a doctor if it gets worse. MODERATE means see a "
    "doctor or the nearest PHC within 24 hours. HIGH means act immediately (call 108/112 or go to "
    "the hospital).\n"
    "8. Keep every reply under 120 words, kind, and respectful. Never blame, judge, or scare the "
    "user.\n"
    "9. If the user asks about anything other than health, politely say you can only help with "
    "health questions and gently guide them back.\n"
    "10. End EVERY reply with exactly this line as the very last line: I give preliminary "
    "guidance only. I am not a doctor."
)

EMERGENCY_HINTS = (
    "chest pain", "chest pressure", "breathing", "breathe", "breathless",
    "unconscious", "fainted", "heavy bleeding", "bleeding a lot",
    "seizure", "fits", "convulsion", "stroke", "poison", "overdose",
    "snakebite", "snake bite", "self-harm", "self harm", "suicide",
    "suicidal", "kill myself", "end my life", "want to die",
    "सांस", "छाती", "बेहोश", "दौरा", "दौरे", "झटके", "लकवा",
    "ज़हर", "जहर", "सांप", "आत्महत्या", "खुदकुशी", "मरना चाहता",
)

EMERGENCY_REMINDER = (
    "\n\nIMPORTANT FOR THIS REPLY: The user's message may describe emergency signs (including "
    "self-harm). If the conversation shows a possible emergency, you MUST put the emergency "
    "instruction FIRST: call 108 or 112 right now for the free ambulance, or go straight to the "
    "nearest hospital (in the user's language), and mark it as 'Urgency: HIGH (उच्च)'. You may "
    "add one or two short safety steps, then end with the required last line. Skip this only if "
    "the message is clearly not an emergency."
)
DOCTOR_MAP = [
    (["pregnan", "garbh", "गर्भ"], "Gynaecologist"),
    (["child", "baby", "bachch", "bacche", "बच्च"], "Paediatrician (child doctor)"),
    (["chest", "seene", "heart", "सीने", "दिल"], "Cardiologist (heart doctor)"),
    (["headache", "sar dard", "sir dard", "सिर"], "Physician or neurologist"),
    (["skin", "rash", "khujli", "त्वचा", "खुजली"], "Dermatologist (skin doctor)"),
    (["ear", "kaan", "throat", "gala", "nose", "naak", "कान", "गला", "नाक"], "ENT doctor"),
    (["eye", "aankh", "आँख", "आंख"], "Eye specialist"),
    (["tooth", "teeth", "daant", "दांत"], "Dentist"),
]

def doctor_line(user_text: str, reply: str) -> str:
    t = user_text.lower()
    doctor = "General physician"
    for keys, name in DOCTOR_MAP:
        if any(k in t for k in keys):
            doctor = name
            break
    r = reply.upper()
    if "HIGH" in r or "उच्च" in reply:
        when = "Go to the nearest hospital emergency now. Call 108 or 112."
    elif "MODERATE" in r or "मध्यम" in reply:
        when = "Visit the nearest PHC or doctor within 1-2 days."
    else:
        when = "See a doctor only if it continues for 2-3 days."
    return f"\n\nWho to see: {doctor}. {when}"
HOSPITALS = [
    # (words the user may type for their city or district, list of hospitals)
    (["cityname", "शहर"], [
        ("HOSPITAL NAME", "PHONE NUMBER", "Area"),
        ("HOSPITAL NAME 2", "PHONE NUMBER", "Area"),
    ]),
    (["cityname2", "शहर2"], [
        ("HOSPITAL NAME", "PHONE NUMBER", "Area"),
    ]),
]

def hospital_text(all_user_text: str) -> str:
    t = all_user_text.lower()
    for keys, hospitals in HOSPITALS:
        if any(k in t for k in keys):
            lines = [f"- {n}, {a}: {p}" for n, p, a in hospitals]
            return "\n\nNearby hospitals:\n" + "\n".join(lines) + "\nAmbulance: 108 or 112."
    return (
        "\n\nTell me your city or district and I will share nearby hospitals. "
        "अपना शहर या ज़िला बताइए। Ambulance / एम्बुलेंस: 108 or 112."
    )

def is_possible_emergency(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in EMERGENCY_HINTS)

# ---------------------------------------------------------------- llm client

client = OpenAI(
    # Agentverse injects both variables into every hosted agent; no key is ever needed here.
    base_url=os.environ["ASI1_BASE_URL"],
    api_key=os.environ["ASI1_API_KEY"],
)

# ------------------------------------------------------------- chat protocol

agent = Agent()
protocol = Protocol(spec=chat_protocol_spec)


def create_text_chat(text: str) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    return ChatMessage(timestamp=datetime.utcnow(), msg_id=uuid4(), content=content)


@protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    # Always acknowledge the incoming message first.
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )

    text = msg.text()
    if not text:
        return

    system_content = SYSTEM_PROMPT
    if is_possible_emergency(text):
        system_content += EMERGENCY_REMINDER

    messages = [
        {"role": "system", "content": system_content},
        *build_llm_message_history(ctx),
    ]

    try:
        completion = client.chat.completions.create(
            model="asi1",
            messages=messages,
            max_tokens=1024,
        )
        reply = str(completion.choices[0].message.content or "")
    except Exception:
        ctx.logger.exception("Error querying the model")
        reply = (
            "Sorry, there was a technical problem. Please try again in a little while. "
            "In an emergency, call 108 or 102 right away.\n\n"
            "क्षमा करें, तकनीकी समस्या आ गई है। कृपया थोड़ी देर बाद दोबारा कोशिश करें। "
            "आपात स्थिति में तुरंत 108 या 102 पर कॉल करें।\n\n"
            "I give preliminary guidance only. I am not a doctor."
        )

    await ctx.send(sender, create_text_chat(reply))


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
