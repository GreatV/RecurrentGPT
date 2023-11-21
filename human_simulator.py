from utils import get_content_between_a_b, parse_instructions, get_api_response


class Human:
    def __init__(self, input, memory, embedder):
        self.input = input
        if memory:
            self.memory = memory
        else:
            self.memory = self.input["output_memory"]
        self.embedder = embedder
        self.output = {}

    def prepare_input(self):
        previous_paragraph = self.input["input_paragraph"]
        writer_new_paragraph = self.input["output_paragraph"]
        memory = self.input["output_memory"]
        user_edited_plan = self.input["output_instruction"]

        input_text = f"""
    现在想象一下，你是一位小说家，在 AI 的帮助下撰写一部中篇小说。你将得到一个你以前写的段落（你自己写的），一个由你的 AI 助手写的段落，一个由你的 AI 助手维护的主要故事情节摘要，以及一个由你的 AI 助手提出的下一步写作计划。
    我需要你
        1. 扩展的段落: 将 AI 助手所写的新段落扩展为你的 AI 助手所写段落长度的两倍。
        2. 选定的计划: 复制 AI 助手提出的计划。
        3. 修订的计划: 将选定的计划修改为下一段的大纲。
    
    以前写过的段落:
    {previous_paragraph}

    由你的 AI 助手维护的主要故事情节摘要:
    {memory}

    由你的 AI 助手撰写的新段落:
    {writer_new_paragraph}

    你的 AI 助手提出的下一步写作计划:
    {user_edited_plan}

    在开始写作，严格按照以下输出格式组织你的输出，所有输出仍然保持是中文:
    
    扩展的段落:
    <输出段落字符串>，约 40-50 句。

    选定的计划: 
    <在此处复制计划>。

    修订的计划:
    <修订计划字符串>，保持简短，约 5-7 句。

    非常重要:
    记住，你是在写小说。要像小说家一样写作，在写下一段计划时，速度不要太快。在选择和扩展计划时，要考虑计划如何才能吸引普通读者。切记遵守篇幅限制！请记住，本章将包含 10 多个段落，而小说将包含 100 多个章节。而下一段将是第二章的第二段。你需要为未来的故事留出空间。

    """
        return input_text

    def parse_plan(self, response):
        plan = get_content_between_a_b("选定的计划:", "理由:", response)
        return plan

    def select_plan(self, response_file):
        previous_paragraph = self.input["input_paragraph"]
        writer_new_paragraph = self.input["output_paragraph"]
        memory = self.input["output_memory"]
        previous_plans = self.input["output_instruction"]
        prompt = f"""
    现在想象一下，你是一个帮助小说家做决策的助手。你将得到一个之前写好的段落和一个由 AI 写作助手写好的段落、一个由 AI 助手维护的主要故事情节摘要，以及 3 个可能的下一步写作计划。
    我需要你:
    选择 AI 助手提出的最有趣、最合适的计划。

    以前写过的段落:  
    {previous_paragraph}

    由你的 AI 助手维护的主要故事情节摘要:
    {memory}

    由你的 AI 助手撰写的新段落:
    {writer_new_paragraph}

    你的 AI 助手提出的三个下一步写作计划:
    {parse_instructions(previous_plans)}

    现在开始选择，严格按照下面的输出格式组织你的输出:
      
    选定的计划: 
    <在此处复制所选计划>

    理由:
    <解释你选择该计划的原因>
    """
        print(prompt + "\n" + "\n")

        response = get_api_response(prompt)

        plan = self.parse_plan(response)
        while plan is None:
            response = get_api_response(prompt)
            plan = self.parse_plan(response)

        if response_file:
            with open(response_file, "a", encoding="utf-8") as f:
                f.write(f"选定的计划:\n{response}\n\n")

        return plan

    def parse_output(self, text):
        try:
            if text.splitlines()[0].startswith("扩展的段落"):
                new_paragraph = get_content_between_a_b("扩展的段落:", "选定的计划", text)
            else:
                new_paragraph = text.splitlines()[0]

            lines = text.splitlines()
            if lines[-1] != "\n" and lines[-1].startswith("修订的计划:"):
                revised_plan = lines[-1][len("修订的计划:") :]
            elif lines[-1] != "\n":
                revised_plan = lines[-1]

            output = {
                "output_paragraph": new_paragraph,
                "output_instruction": revised_plan,
            }

            return output
        except Exception as e:
            print(e)
            return None

    def step(self, response_file=None):
        prompt = self.prepare_input()
        print(prompt + "\n" + "\n")

        response = get_api_response(prompt)
        self.output = self.parse_output(response)
        while self.output is None:
            response = get_api_response(prompt)
            self.output = self.parse_output(response)
        if response_file:
            with open(response_file, "a", encoding="utf-8") as f:
                f.write(f"人类输出:\n{response}\n\n")
