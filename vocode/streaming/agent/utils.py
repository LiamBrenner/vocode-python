from copy import deepcopy
import re
from typing import (
    Dict,
    Any,
    AsyncGenerator,
    AsyncIterable,
    Callable,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)

from vocode.streaming.models.actions import FunctionCall, FunctionFragment
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import (
    ActionFinish,
    ActionStart,
    EventLog,
    Message,
    Transcript,
)

SENTENCE_ENDINGS = [".", "!", "?", "\n"]


async def collate_response_async(
    gen: AsyncIterable[Union[str, FunctionFragment]],
    sentence_endings: List[str] = SENTENCE_ENDINGS,
    get_functions: Literal[True, False] = False,
) -> AsyncGenerator[Union[str, FunctionCall], None]:
    print("collate_response_async")
    print(f"collate_response_async: gen is {gen}")
    sentence_endings_pattern = "|".join(map(re.escape, sentence_endings))
    list_item_ending_pattern = r"\n"
    buffer = ""
    function_name_buffer = ""
    function_args_buffer = ""
    prev_ends_with_money = False
    async for token in gen:
        print("collate_response_async: token is - ", token)
        if not token:
            print("collate_response_async: no token, continuing")
            continue
        if isinstance(token, str):
            print("collate_response_async: token is of type str")
            if prev_ends_with_money and token.startswith(" "):
                yield buffer.strip()
                buffer = ""

            buffer += token
            possible_list_item = bool(re.match(r"^\d+[ .]", buffer))
            ends_with_money = bool(re.findall(r"\$\d+.$", buffer))
            if re.findall(
                list_item_ending_pattern
                if possible_list_item
                else sentence_endings_pattern,
                token,
            ):
                if not ends_with_money:
                    to_return = buffer.strip()
                    if to_return:
                        yield to_return
                    buffer = ""
            prev_ends_with_money = ends_with_money
        elif isinstance(token, FunctionFragment):
            function_name_buffer += token.name
            function_args_buffer += token.arguments
    to_return = buffer.strip()
    print(f"collate_response_async: response is - {to_return}")
    if to_return:
        print(f"collate_response_async: there is a to return value, yay!")
        yield to_return
    if function_name_buffer and get_functions:
        yield FunctionCall(name=function_name_buffer, arguments=function_args_buffer)


async def openai_get_tokens(gen) -> AsyncGenerator[Union[str, FunctionFragment], None]:
    print("openai_get_tokens")
    async for event in gen:
        print(f"openai_get_tokens: event in gen is {event}")
        choices = event.choices
        if len(choices) == 0:
            print("openai_get_tokens: length of choices is 0, continuing")
            continue
        choice = choices[0]
        if choice.finish_reason:
            print("openai_get_tokens: finish reason detected, breaking")
            break
        delta = choice.delta
        # if delta.text:
        #     token = delta.text
        #     print(f"openai_get_tokens: text in delta and not none, yielding: {token}")
        #     yield token
        if delta.content:
            token = delta.content
            print(f"openai_get_tokens: content in delta and not none, yielding: {token}")
            yield token
        elif delta.function_call:
            print(f"openai_get_tokens: function in delta and not none, yielding a function call")
            yield FunctionFragment(
                name=delta["function_call"]["name"]
                if "name" in delta["function_call"]
                else "",
                arguments=delta["function_call"]["arguments"]
                if "arguments" in delta["function_call"]
                else "",
            )


def find_last_punctuation(buffer: str) -> Optional[int]:
    print("find_last_punctuation")
    indices = [buffer.rfind(ending) for ending in SENTENCE_ENDINGS]
    if not indices:
        return None
    return max(indices)


def get_sentence_from_buffer(buffer: str):
    print("get_sentence_from_buffer")
    last_punctuation = find_last_punctuation(buffer)
    if last_punctuation:
        return buffer[: last_punctuation + 1], buffer[last_punctuation + 1 :]
    else:
        return None, None


def format_openai_chat_messages_from_transcript(
    transcript: Transcript, prompt_preamble: Optional[str] = None
) -> List[dict]:
    print("format_openai_chat_messages_from_transcript")
    chat_messages: List[Dict[str, Optional[Any]]] = (
        [{"role": "system", "content": prompt_preamble}] if prompt_preamble else []
    )

    # merge consecutive bot messages
    new_event_logs: List[EventLog] = []
    idx = 0
    while idx < len(transcript.event_logs):
        bot_messages_buffer: List[Message] = []
        current_log = transcript.event_logs[idx]
        while isinstance(current_log, Message) and current_log.sender == Sender.BOT:
            bot_messages_buffer.append(current_log)
            idx += 1
            try:
                current_log = transcript.event_logs[idx]
            except IndexError:
                break
        if bot_messages_buffer:
            merged_bot_message = deepcopy(bot_messages_buffer[-1])
            merged_bot_message.text = " ".join(
                event_log.text for event_log in bot_messages_buffer
            )
            new_event_logs.append(merged_bot_message)
        else:
            new_event_logs.append(current_log)
            idx += 1

    for event_log in new_event_logs:
        if isinstance(event_log, Message):
            chat_messages.append(
                {
                    "role": "assistant" if event_log.sender == Sender.BOT else "user",
                    "content": event_log.text,
                }
            )
        elif isinstance(event_log, ActionStart):
            chat_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": event_log.action_type,
                        "arguments": event_log.action_input.params.json(),
                    },
                }
            )
        elif isinstance(event_log, ActionFinish):
            chat_messages.append(
                {
                    "role": "function",
                    "name": event_log.action_type,
                    "content": event_log.action_output.response.json(),
                }
            )
    return chat_messages


def vector_db_result_to_openai_chat_message(vector_db_result):
    return {"role": "user", "content": vector_db_result}
