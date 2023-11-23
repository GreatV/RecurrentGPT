import os
import re
from langchain.chat_models import QianfanChatEndpoint
from langchain.schema import HumanMessage, SystemMessage


def get_api_response(content: str, max_tokens=None):
    chat = QianfanChatEndpoint(
        model="ERNIE-Bot",
        qianfan_ak=os.environ["QIANFAN_API_KEY"],
        qianfan_sk=os.environ["QIANFAN_SECRET_KEY"],
    )

    messages = [
        SystemMessage(content="你是一位富有创造力和想象力的小说写作助手。"),
        HumanMessage(content=content),
    ]
    try:
        response = chat(messages)
    except Exception as e:
        print(e)
        raise e

    if response is not None:
        return response.content
    else:
        return "Error: response not found"


def get_content_between_a_b(a, b, text):
    try:
        text = text.replace("：", ":")
        text = re.search(f"{a}(.*?)\n{b}", text, re.DOTALL).group(1).strip()
        text = text.replace(":", "：")
    except Exception as e:
        text = ""

    return text


def get_init(init_text=None, text=None, response_file=None):
    """
    init_text: if the title, outline, and the first 3 paragraphs are given in a .txt file, directly read
    text: if no .txt file is given, use init prompt to generate
    """
    if not init_text:
        response = get_api_response(text)
        print(response)

        if response_file:
            with open(response_file, "a", encoding="utf-8") as f:
                f.write(f"初始输出:\n{response}\n\n")
    else:
        with open(init_text, "r", encoding="utf-8") as f:
            response = f.read()
        f.close()
    paragraphs = {
        "name": "",
        "Outline": "",
        "Paragraph 1": "",
        "Paragraph 2": "",
        "Paragraph 3": "",
        "Summary": "",
        "Instruction 1": "",
        "Instruction 2": "",
        "Instruction 3": "",
    }
    paragraphs["name"] = get_content_between_a_b("名称:", "大纲", response)

    paragraphs["Paragraph 1"] = get_content_between_a_b("段落 1:", "段落 2", response)
    paragraphs["Paragraph 2"] = get_content_between_a_b("段落 2:", "段落 3", response)
    paragraphs["Paragraph 3"] = get_content_between_a_b("段落 3:", "情节摘要", response)
    paragraphs["Summary"] = get_content_between_a_b("情节摘要:", "指令 1", response)
    paragraphs["Instruction 1"] = get_content_between_a_b("指令 1:", "指令 2", response)
    paragraphs["Instruction 2"] = get_content_between_a_b("指令 2:", "指令 3", response)
    lines = response.splitlines()
    # content of Instruction 3 may be in the same line with I3 or in the next line
    if lines[-1] != "\n" and lines[-1].startswith("指令 3"):
        paragraphs["Instruction 3"] = lines[-1][len("指令 3:") :]
    elif lines[-1] != "\n":
        paragraphs["Instruction 3"] = lines[-1]
    # Sometimes it gives Chapter outline, sometimes it doesn't
    for line in lines:
        if line.startswith("章节"):
            paragraphs["Outline"] = get_content_between_a_b("大纲:", "章节", response)
            break
    if paragraphs["Outline"] == "":
        paragraphs["Outline"] = get_content_between_a_b("大纲:", "段落", response)

    return paragraphs


def get_chatgpt_response(model, prompt):
    response = ""
    for data in model.ask(prompt):
        response = data["message"]
    model.delete_conversation(model.conversation_id)
    model.reset_chat()
    return response


def parse_instructions(instructions):
    output = ""
    for i in range(len(instructions)):
        output += f"{i+1}. {instructions[i]}\n"
    return output
