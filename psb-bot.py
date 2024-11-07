import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import json
import logging
from typing import Literal, Optional

import discord
import httpx
from openai import AsyncOpenAI
import ollama
import chromadb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gpt-4o", "claude-3", "gemini", "pixtral", "llava", "vision")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

ALLOWED_FILE_TYPES = ("image", "text")
ALLOWED_CHANNEL_TYPES = (discord.ChannelType.text, discord.ChannelType.public_thread, discord.ChannelType.private_thread, discord.ChannelType.private)

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 100

EMBEDDINGS_MODEL = "aroxima/gte-qwen2-1.5b-instruct:latest"


def get_config(filename="config.json"):
    """ reading configuration """
    with open(filename, "r", encoding='utf-8') as file:
        return {k: v for d in json.load(file).values() for k, v in d.items()}


async def get_embeddings(text:str):
    """ embeddings calculated from the prompt """
    return ollama.embeddings(model=EMBEDDINGS_MODEL, prompt=text)['embedding']


async def get_results(q_embeddings, num_of_results=5):
    """ vector database search """
    return  collection.query(query_embeddings=[q_embeddings],
                             n_results=num_of_results,
                             include=["documents", "metadatas", "distances"])


chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_collection(name="psb")

cfg = get_config()

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=cfg["status_message"][:128] or "github.com/jakobdylanc/llmcord.py")
discord_client = discord.Client(intents=intents, activity=activity)

httpx_client = httpx.AsyncClient()

msg_nodes = {}
last_task_time = None

if cfg["client_id"] != 123456789:
    print(f"\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={cfg['client_id']}&permissions=412317273088&scope=bot\n")


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    next_msg: Optional[discord.Message] = None

    has_bad_attachments: bool = False
    fetch_next_failed: bool = False

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time

    if (
        new_msg.channel.type not in ALLOWED_CHANNEL_TYPES
        or (new_msg.channel.type != discord.ChannelType.private and discord_client.user not in new_msg.mentions)
        or new_msg.author.bot
    ):
        return

    cfg = get_config()

    if (cfg["allowed_channel_ids"] and not any(id in cfg["allowed_channel_ids"] for id in (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None)))) or (
        cfg["allowed_role_ids"] and (new_msg.channel.type == discord.ChannelType.private or not any(role.id in cfg["allowed_role_ids"] for role in new_msg.author.roles))
    ):
        return

    provider, model = cfg["model"].split("/", 1)
    base_url = cfg["providers"][provider]["base_url"]
    api_key = cfg["providers"][provider].get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    accept_images: bool = any(x in model for x in VISION_MODEL_TAGS)
    accept_usernames: bool = any(x in provider for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = cfg["max_text"]
    max_images = cfg["max_images"] if accept_images else 0
    max_messages = cfg["max_messages"]

    use_plain_responses: bool = cfg["use_plain_responses"]
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg
    sources = "" # źródła znalezione przez RAG
    while curr_msg and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                good_attachments = {type: [att for att in curr_msg.attachments if att.content_type and type in att.content_type] for type in ALLOWED_FILE_TYPES}

                # RAG
                if curr_msg.content:
                    query = curr_msg.content
                    query = query.strip()
                    if query.startswith('<'):
                        pos = query.find('>')
                        query = query[pos+1:].strip()
                    embeded_query = await get_embeddings(text=query)
                    results = await get_results(q_embeddings=embeded_query, num_of_results=5)
                    logging.info(results["distances"][0])
                    documents = results["documents"][0]
                    sources_list = []
                    for mt in results["metadatas"][0]:
                        if mt["book"] == "Strona internetowa PSB":
                            t_source = f'"{mt["biogram"]}", {mt["book"]} ({mt["publication_year"]})'
                        else:
                            t_source = f'"{mt["biogram"]}", {mt["book"]}, tom {mt["volume"]}, {mt["page"]} ({mt["publication_year"]})'
                        if t_source not in sources_list:
                            sources_list.append(t_source)
                    if sources_list:
                        s_counter = 0
                        for source_item in sources_list:
                            s_counter += 1
                            sources += f'{s_counter}. {source_item}\n'
                        sources = '\n\nPrzeanalizowane źródła:\n\n' + sources

                curr_node.text = "\n".join(
                    ([curr_msg.content] if curr_msg.content else [])
                    + [embed.description for embed in curr_msg.embeds if embed.description]
                    + [document + "\n" for document in documents]
                    + [(await httpx_client.get(att.url)).text for att in good_attachments["text"]]
                )
                if curr_node.text.startswith(discord_client.user.mention):
                    curr_node.text = curr_node.text.replace(discord_client.user.mention, "", 1).lstrip()

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode((await httpx_client.get(att.url)).content).decode('utf-8')}"))
                    for att in good_attachments["image"]
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_client.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > sum(len(att_list) for att_list in good_attachments.values())

                try:
                    if (
                        curr_msg.reference == None
                        and curr_msg.channel.type != discord.ChannelType.private
                        and discord_client.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and any(prev_msg_in_channel.type == type for type in (discord.MessageType.default, discord.MessageType.reply))
                        and prev_msg_in_channel.author == curr_msg.author
                    ):
                        curr_node.next_msg = prev_msg_in_channel
                    else:
                        next_is_thread_parent: bool = curr_msg.reference == None and curr_msg.channel.type == discord.ChannelType.public_thread
                        if next_msg_id := curr_msg.channel.id if next_is_thread_parent else getattr(curr_msg.reference, "message_id", None):
                            next_node = msg_nodes.setdefault(next_msg_id, MsgNode())
                            while next_node.lock.locked():
                                await asyncio.sleep(0)

                            curr_node.next_msg = (
                                (curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(next_msg_id))
                                if next_is_thread_parent
                                else (curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(next_msg_id))
                            )
                except (discord.NotFound, discord.HTTPException, AttributeError):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_next_failed = True

            if curr_node.text[:max_text] or curr_node.images[:max_images]:
                if accept_images and curr_node.images[:max_images]:
                    content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
                else:
                    content = curr_node.text[:max_text]

                message = dict(role=curr_node.role, content=content)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_next_failed or (curr_node.next_msg and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.next_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if cfg["system_prompt"]:
        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        if accept_usernames:
            system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")

        system_prompt = dict(role="system", content="\n".join([cfg["system_prompt"]] + system_prompt_extras))
        messages.append(system_prompt)

    # Generate and send response message(s) (can be multiple if response is long)
    response_msgs = []
    response_contents = []
    prev_chunk = None
    edit_task = None

    kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_body=cfg["extra_api_parameters"])
    try:
        async with new_msg.channel.typing():
            async for curr_chunk in await openai_client.chat.completions.create(**kwargs):
                prev_content = prev_chunk.choices[0].delta.content if prev_chunk != None and prev_chunk.choices[0].delta.content else ""
                curr_content = curr_chunk.choices[0].delta.content or ""

                if response_contents or prev_content:
                    if response_contents == [] or len(response_contents[-1] + prev_content) > max_message_length:
                        response_contents.append("")

                        if not use_plain_responses:
                            embed = discord.Embed(description=(prev_content + STREAMING_INDICATOR), color=EMBED_COLOR_INCOMPLETE)
                            for warning in sorted(user_warnings):
                                embed.add_field(name=warning, value="", inline=False)

                            reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                            msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()
                            response_msgs.append(response_msg)
                            last_task_time = dt.now().timestamp()

                    response_contents[-1] += prev_content

                    if not use_plain_responses:
                        finish_reason = curr_chunk.choices[0].finish_reason

                        ready_to_edit: bool = (edit_task == None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                        msg_split_incoming: bool = len(response_contents[-1] + curr_content) > max_message_length
                        is_final_edit: bool = finish_reason != None or msg_split_incoming
                        is_good_finish: bool = finish_reason != None and any(finish_reason.lower() == x for x in ("stop", "end_turn"))

                        if ready_to_edit or is_final_edit:
                            while edit_task != None and not edit_task.done():
                                await asyncio.sleep(0)

                            # jeżeli model nie może odpowiedzieć na pytanie na podstawie kontekstu
                            # o podawanie listy źródeł nie jest potrzebne.
                            if response_contents[-1] and 'niestety, nie mam informacji na ten temat' in response_contents[-1].lower():
                                sources = ''

                            embed.description = response_contents[-1] + sources if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE
                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                            last_task_time = dt.now().timestamp()

                prev_chunk = curr_chunk

        if use_plain_responses:
            for content in response_contents:
                reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                response_msg = await reply_to_msg.reply(content=content)
                msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                await msg_nodes[response_msg.id].lock.acquire()
                response_msgs.append(response_msg)



    except:
        logging.exception("Error while generating response")

    for msg in response_msgs:
        msg_nodes[msg.id].text = "".join(response_contents)
        msg_nodes[msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main():
    await discord_client.start(cfg["bot_token"])


asyncio.run(main())