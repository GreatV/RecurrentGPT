from utils import get_content_between_a_b, get_api_response
import paddle

import random

from paddlenlp.embeddings import TokenEmbedding

token_embedding = TokenEmbedding("w2v.baidu_encyclopedia.target.word-word.dim300")


class RecurrentGPT:
    def __init__(self, input, short_memory, long_memory):
        self.input = input
        self.short_memory = short_memory
        self.long_memory = long_memory
        self.output = {}

    def prepare_input(self, new_character_prob=0.1, top_k=2):
        input_paragraph = self.input["output_paragraph"]
        input_instruction = self.input["output_instruction"]

        # get the top 3 most similar paragraphs from memory
        memory_scores = []
        for memory in self.long_memory:
            memory_scores.append(token_embedding.cosine_sim(input_instruction, memory))
        memory_scores = paddle.to_tensor(memory_scores)
        top_k_idx = paddle.topk(memory_scores, k=top_k)[1]
        top_k_memory = [self.long_memory[idx] for idx in top_k_idx]
        # combine the top 3 paragraphs
        input_long_term_memory = "\n".join(
            [
                f"相关段落 {i+1} :" + selected_memory
                for i, selected_memory in enumerate(top_k_memory)
            ]
        )
        # randomly decide if a new character should be introduced
        if random.random() < new_character_prob:
            new_character_prompt = "如果合理，可以在输出段落中引入新角色，并将其添加到记忆中。"
        else:
            new_character_prompt = ""

        input_text = f"""我需要你帮我写一部小说。现在我给你一个 400 字的记忆（简短摘要），你应该用它来存储已写内容的关键部分，这样你就能掌握很长的上下文。每次，我都会给你当前的记忆（以前故事的简短摘要。你应该用它来存储已写内容的关键内容，以便你能跟踪很长的上下文）、之前写的段落以及下一段要写什么的说明。
    我需要你写如下部分：
    1. 输出段落: 小说的下一段。输出段落应包含约 20 个句子，并应遵循输入说明。
    2. 输出记忆: 更新后的记忆。你应首先解释输入记忆中哪些句子不再需要以及原因，然后解释记忆中需要添加哪些内容以及原因。然后再写出更新后的记忆体。更新后的记忆应与输入记忆类似，但你之前认为应删除或添加的部分除外。更新的记忆应只存储关键信息。更新记忆的句子不得超过 20 句！
    3. 输出指令: 说明下一步要写什么（在你所写的内容之后）。你应该输出 3 个不同的指令，每个指令都可能是故事有趣的延续。每个输出指令应包含约 5 个句子
    以下是输入内容:
    输入记忆:
    {self.short_memory}

    输入段落:
    {input_paragraph}

    输入指令:
    {input_instruction}

    输入相关段落:
    {input_long_term_memory}
    
    现在开始编写，严格按照以下输出格式组织输出：
    输出段落:
        <输出段落字符串>，约 20 句。

    输出记忆:
        合理性: <说明如何更新内存的字符串>;
        更新的记忆: <更新记忆的字符串>，大约 10 到 20 句话

    输出指令:
        指令 1: <指令 1 的内容>，约 5 句话
        指令 2: <指令 2 的内容>，约 5 句话
        指令 3: <指令 3 的内容>，约 5 句话

    非常重要 更新的记忆只能存储关键信息。更新记忆的字数绝不能超过 500 字！
    最后，请记住你正在写的是一部小说。要像小说家一样写作，在编写下一段的输出指令时，速度不要太快。请记住，本章将包含 10 多个段落，而小说将包含 100 多个章节。这仅仅是个开始。写一些接下来会发生的有趣的人事即可。此外，在编写输出说明时，还要考虑哪些情节能够吸引普通读者。

    非常重要： 
    首先要解释输入记忆中哪些句子不再需要，为什么，然后解释记忆中需要添加哪些内容，为什么。然后，开始重写输入记忆，得到更新后的记忆。
    {new_character_prompt}
    """
        return input_text

    def parse_output(self, output):
        try:
            output_paragraph = get_content_between_a_b("输出段落:", "输出记忆", output)
            output_memory_updated = get_content_between_a_b("更新的记忆:", "输出指令", output)
            self.short_memory = output_memory_updated
            ins_1 = get_content_between_a_b("指令 1:", "指令 2", output)
            ins_2 = get_content_between_a_b("指令 2:", "指令 3", output)
            lines = output.splitlines()
            # content of Instruction 3 may be in the same line with I3 or in the next line
            if lines[-1] != "\n" and lines[-1].startswith("指令 3"):
                ins_3 = lines[-1][len("指令 3:") :]
            elif lines[-1] != "\n":
                ins_3 = lines[-1]

            output_instructions = [ins_1, ins_2, ins_3]
            assert len(output_instructions) == 3

            output = {
                "input_paragraph": self.input["output_paragraph"],
                "output_memory": output_memory_updated,  # feed to human
                "output_paragraph": output_paragraph,
                "output_instruction": [
                    instruction.strip() for instruction in output_instructions
                ],
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
                f.write(f"作者输出:\n{response}\n\n")

        self.long_memory.append(self.input["output_paragraph"])
