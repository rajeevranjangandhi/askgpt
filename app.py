from agent import get_agent
from langchain_core.messages import SystemMessage, HumanMessage
from database import init_db
from tools import set_current_thread_id

init_db()  # create tables (conversations, chat_messages, long_term_memory) if missing

agent = get_agent("gemini-3.8-flash")

config = {
    "configurable":{
        "thread_id":"test_thread_id"
    }
}
set_current_thread_id(config["configurable"]["thread_id"])  # memory tools use the same thread

for message_chunk, metadata in agent.stream(
    {"messages":[HumanMessage(content="My name is Rajeev, have you remember that")]},config=config,stream_mode="messages"):
    if message_chunk.content:
        print(message_chunk.content,end="",flush=True)
