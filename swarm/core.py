# Standard library imports
import copy
import json
from collections import defaultdict
from typing import List, Callable, Union

# Package/library imports
from openai import OpenAI


# Local imports
from .util import function_to_json, debug_print, merge_chunk
from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
)

__CTX_VARS_NAME__ = "context_variables"


class Swarm:
    def __init__(self, client=None):
        if not client:
            client = OpenAI()
        self.client = client

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
        extra_completion_params: dict = {},
    ) -> ChatCompletionMessage:
        """Get a chat completion from the OpenAI API.

        Args:
            agent (Agent): The agent object.
            history (List): The history of messages.
            context_variables (dict): The context variables.
            model_override (str): The model override.
            stream (bool): Whether to stream the response.
            debug (bool): Whether to print debug information.
            extra_completion_params (dict, optional): Additional parameters to pass to the OpenAI chat completion API. ref:https://platform.openai.com/docs/api-reference/chat/create
                Common options include:
                - temperature (float): What sampling temperature to use, between 0 and 2.
                  Higher values like 0.8 will make the output more random, while lower values
                  like 0.2 will make it more focused and deterministic.
                - top_p (float): An alternative to sampling with temperature, called nucleus sampling.
                  Range is between 0 and 1. We generally recommend altering this or temperature but not both.
                - n (int): How many chat completion choices to generate for each input message.
                  Note that the API may return fewer choices if some were flagged.
                - max_tokens (int): The maximum number of tokens to generate in the chat completion.
                - presence_penalty (float): Number between -2.0 and 2.0. Positive values penalize new tokens
                  based on whether they appear in the text so far, increasing the model's likelihood
                  to talk about new topics.
                - frequency_penalty (float): Number between -2.0 and 2.0. Positive values penalize new tokens
                  based on their existing frequency in the text so far, decreasing the model's likelihood
                  to repeat the same line verbatim.
                - seed (int): An integer between 0 and 2^32-1 used for deterministic sampling.
                  When provided along with identical parameters (messages, temperature, etc.),
                  the API will make a best effort to return the same output. This is useful for:
                  * Reproducible results in testing
                  * Consistent behavior across multiple API calls
                  * A/B testing different prompts with the same randomness
                  Note: This is a best-effort feature and does not guarantee identical outputs.
                - response_format (dict): An object specifying the format that the model must output.
                  Setting to {"type": "json_object"} enables JSON mode, which guarantees the message the
                  model generates is valid JSON.
                - stop (string|array): Up to 4 sequences where the API will stop generating further tokens.

                Example:
                    ```python
                    swarm.run(
                        agent=my_agent,
                        messages=messages,
                        extra_completion_params={
                            "temperature": 0.7,           # 适度的随机性
                            "max_tokens": 1000,          # 限制回复长度
                            "presence_penalty": 0.6,     # 鼓励谈论新话题
                            "frequency_penalty": 0.6,    # 减少重复
                            "response_format": {         # 强制返回JSON格式
                                "type": "json_object"
                            },
                            "seed": 123                  # 确保结果可重现
                        }
                    )
                    ```

        Returns:
            ChatCompletionMessage: The completion response from the API.
        """
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history
        debug_print(debug, "Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        # 防止关键参数被覆盖
        protected_keys = {"model", "messages", "tools", "tool_choice", "stream"}
        filtered_params = {
            k: v for k, v in extra_completion_params.items() if k not in protected_keys
        }

        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "stream": stream,
            **filtered_params,
        }

        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        return self.client.chat.completions.create(**create_params)

    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    debug_print(debug, error_message)
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        context_variables: dict,
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in function_map:
                debug_print(debug, f"Tool {name} not found in function map.")
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            debug_print(debug, f"Processing tool call: {name} with arguments {args}")

            func = function_map[name]
            # pass context_variables to agent functions
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = function_map[name](**args)

            result: Result = self.handle_function_result(raw_result, debug)
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": result.value,
                }
            )
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
        extra_completion_params: dict = {},
    ):
        """
        New feature: extra_completion_params
        extra_completion_params (dict, optional): Additional parameters to pass to the OpenAI chat completion API. ref:https://platform.openai.com/docs/api-reference/chat/create
                Common options include:
                - temperature (float): What sampling temperature to use, between 0 and 2.
                  Higher values like 0.8 will make the output more random, while lower values
                  like 0.2 will make it more focused and deterministic.
                - top_p (float): An alternative to sampling with temperature, called nucleus sampling.
                  Range is between 0 and 1. We generally recommend altering this or temperature but not both.
                - n (int): How many chat completion choices to generate for each input message.
                  Note that the API may return fewer choices if some were flagged.
                - max_tokens (int): The maximum number of tokens to generate in the chat completion.
                - presence_penalty (float): Number between -2.0 and 2.0. Positive values penalize new tokens
                  based on whether they appear in the text so far, increasing the model's likelihood
                  to talk about new topics.
                - frequency_penalty (float): Number between -2.0 and 2.0. Positive values penalize new tokens
                  based on their existing frequency in the text so far, decreasing the model's likelihood
                  to repeat the same line verbatim.
                - seed (int): An integer between 0 and 2^32-1 used for deterministic sampling.
                  When provided along with identical parameters (messages, temperature, etc.),
                  the API will make a best effort to return the same output. This is useful for:
                  * Reproducible results in testing
                  * Consistent behavior across multiple API calls
                  * A/B testing different prompts with the same randomness
                  Note: This is a best-effort feature and does not guarantee identical outputs.
                - response_format (dict): An object specifying the format that the model must output.
                  Setting to {"type": "json_object"} enables JSON mode, which guarantees the message the
                  model generates is valid JSON.
                - stop (string|array): Up to 4 sequences where the API will stop generating further tokens.

                Example:
                    ```python
                    swarm.run(
                        agent=my_agent,
                        messages=messages,
                        extra_completion_params={
                            "temperature": 0.7,           # 适度的随机性
                            "max_tokens": 1000,          # 限制回复长度
                            "presence_penalty": 0.6,     # 鼓励谈论新话题
                            "frequency_penalty": 0.6,    # 减少重复
                            "response_format": {         # 强制返回JSON格式
                                "type": "json_object"
                            },
                            "seed": 123                  # 确保结果可重现
                        }
                    )
                    ```
        """
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:

            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
                extra_completion_params=extra_completion_params,
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None
            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
        extra_completion_params: dict = {},
    ) -> Response:
        """
        New feature: extra_completion_params
        extra_completion_params (dict, optional): Additional parameters to pass to the OpenAI chat completion API. ref:https://platform.openai.com/docs/api-reference/chat/create
                Common options include:
                - temperature (float): What sampling temperature to use, between 0 and 2.
                  Higher values like 0.8 will make the output more random, while lower values
                  like 0.2 will make it more focused and deterministic.
                - top_p (float): An alternative to sampling with temperature, called nucleus sampling.
                  Range is between 0 and 1. We generally recommend altering this or temperature but not both.
                - n (int): How many chat completion choices to generate for each input message.
                  Note that the API may return fewer choices if some were flagged.
                - max_tokens (int): The maximum number of tokens to generate in the chat completion.
                - presence_penalty (float): Number between -2.0 and 2.0. Positive values penalize new tokens
                  based on whether they appear in the text so far, increasing the model's likelihood
                  to talk about new topics.
                - frequency_penalty (float): Number between -2.0 and 2.0. Positive values penalize new tokens
                  based on their existing frequency in the text so far, decreasing the model's likelihood
                  to repeat the same line verbatim.
                - seed (int): An integer between 0 and 2^32-1 used for deterministic sampling.
                  When provided along with identical parameters (messages, temperature, etc.),
                  the API will make a best effort to return the same output. This is useful for:
                  * Reproducible results in testing
                  * Consistent behavior across multiple API calls
                  * A/B testing different prompts with the same randomness
                  Note: This is a best-effort feature and does not guarantee identical outputs.
                - response_format (dict): An object specifying the format that the model must output.
                  Setting to {"type": "json_object"} enables JSON mode, which guarantees the message the
                  model generates is valid JSON.
                - stop (string|array): Up to 4 sequences where the API will stop generating further tokens.

                Example:
                    ```python
                    swarm.run(
                        agent=my_agent,
                        messages=messages,
                        extra_completion_params={
                            "temperature": 0.7,           # 适度的随机性
                            "max_tokens": 1000,          # 限制回复长度
                            "presence_penalty": 0.6,     # 鼓励谈论新话题
                            "frequency_penalty": 0.6,    # 减少重复
                            "response_format": {         # 强制返回JSON格式
                                "type": "json_object"
                            },
                            "seed": 123                  # 确保结果可重现
                        }
                    )
                    ```
        """

        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
                extra_completion_params=extra_completion_params,
            )
            message = completion.choices[0].message
            debug_print(debug, "Received completion:", message)
            message.sender = active_agent.name
            history.append(
                json.loads(message.model_dump_json())
            )  # to avoid OpenAI types (?)

            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                message.tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
