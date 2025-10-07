"""
Additional Utilities for ChatSession
"""

import typing

if typing.TYPE_CHECKING:
    from llmchat.chat_session import ChatSession


def get_prompt_message(cs: 'ChatSession') -> str:
    from llmchat.components.model_options import HuggingFaceModelOption

    model_name = cs.config.get('model_option_name', 'unknown')
    output_mode = cs.config.get('output_mode', 'simple')

    if isinstance(cs.model_option, HuggingFaceModelOption):
        device_display = ''
        if device:= cs.config.get('device', None):
            device_display = ' ({})'.format(device)
        return "[{model}{device}][{mode}]>  ".format(model=model_name, device=device_display, mode=output_mode)

    return "[{model}][{mode}]>  ".format(model=model_name, mode=output_mode)
