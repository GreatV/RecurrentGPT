import gradio as gr
import random
from recurrentgpt import RecurrentGPT
from human_simulator import Human
from utils import get_init, parse_instructions

_CACHE = {}


def init_prompt(novel_type, description):
    if description == "":
        description = ""
    else:
        description = " about " + description
    return f"""
请写一篇关于 {description} 的 {novel_type} 小说, 大约50章。请严格按照以下格式:

以小说名称开头。
接下来，写出第一章的大纲。大纲应描述小说的背景和开头。
根据你的提纲写出小说的前三段，并表明小说的内容。用小说的风格写，慢慢地设置场景。
写一个情节摘要，抓住这三段的关键信息。
最后，写出三个不同的下一步写作指令，每个指令包含大约五句话。每条指令都应该是故事可能的、有趣的延续。
输出格式应遵循以下指导原则:
名称: <小说名称>
大纲: <第一章大纲>
段落 1: <第 1 段的内容>
段落 2: <第 2 段的内容>
段落 3: <第 3 段的内容>
情节摘要: <摘要的内容>
指令 1: <指令 1 的内容>
指令 2: <指令 2 的内容>
指令 3: <指令 3 的内容>

确保准确无误，并严格按照输出格式操作。

"""


def init(novel_type, description, request: gr.Request):
    if novel_type == "":
        novel_type = "玄幻"
    global _CACHE
    cookie = "cookie"
    # prepare first init
    init_paragraphs = get_init(text=init_prompt(novel_type, description))
    # print(init_paragraphs)
    start_input_to_human = {
        "output_paragraph": init_paragraphs["Paragraph 3"],
        "input_paragraph": "\n\n".join(
            [init_paragraphs["Paragraph 1"], init_paragraphs["Paragraph 2"]]
        ),
        "output_memory": init_paragraphs["Summary"],
        "output_instruction": [
            init_paragraphs["Instruction 1"],
            init_paragraphs["Instruction 2"],
            init_paragraphs["Instruction 3"],
        ],
    }

    _CACHE[cookie] = {
        "start_input_to_human": start_input_to_human,
        "init_paragraphs": init_paragraphs,
    }
    written_paras = f"""标题: {init_paragraphs['name']}

大纲: {init_paragraphs['Outline']}

段落:

{start_input_to_human['input_paragraph']}"""
    long_memory = parse_instructions(
        [init_paragraphs["Paragraph 1"], init_paragraphs["Paragraph 2"]]
    )
    # short memory, long memory, current written paragraphs, 3 next instructions
    return (
        start_input_to_human["output_memory"],
        long_memory,
        written_paras,
        init_paragraphs["Instruction 1"],
        init_paragraphs["Instruction 2"],
        init_paragraphs["Instruction 3"],
    )


def step(
    short_memory,
    long_memory,
    instruction1,
    instruction2,
    instruction3,
    current_paras,
    request: gr.Request,
):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    cookie = "cookie"
    cache = _CACHE[cookie]

    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human["output_instruction"] = [
            instruction1,
            instruction2,
            instruction3,
        ]
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human, memory=None)
        human.step()
        start_short_memory = init_paragraphs["Summary"]
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(
            input=writer_start_input,
            short_memory=start_short_memory,
            long_memory=[
                init_paragraphs["Paragraph 1"],
                init_paragraphs["Paragraph 2"],
            ],
        )
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output["output_memory"] = short_memory
        # randomly select one instruction out of three
        instruction_index = random.randint(0, 2)
        output["output_instruction"] = [instruction1, instruction2, instruction3][
            instruction_index
        ]
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    long_memory = [[v] for v in writer.long_memory]
    # short memory, long memory, current written paragraphs, 3 next instructions
    return (
        writer.output["output_memory"],
        long_memory,
        current_paras + "\n\n" + writer.output["input_paragraph"],
        human.output["output_instruction"],
        *writer.output["output_instruction"],
    )


def controled_step(
    short_memory,
    long_memory,
    selected_instruction,
    current_paras,
    request: gr.Request,
):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    cookie = "cookie"

    cache = _CACHE[cookie]
    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human["output_instruction"] = selected_instruction
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human, memory=None)
        human.step()
        start_short_memory = init_paragraphs["Summary"]
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(
            input=writer_start_input,
            short_memory=start_short_memory,
            long_memory=[
                init_paragraphs["Paragraph 1"],
                init_paragraphs["Paragraph 2"],
            ],
            memory_index=None,
        )
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output["output_memory"] = short_memory
        output["output_instruction"] = selected_instruction
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    # short memory, long memory, current written paragraphs, 3 next instructions
    return (
        writer.output["output_memory"],
        parse_instructions(writer.long_memory),
        current_paras + "\n\n" + writer.output["input_paragraph"],
        *writer.output["output_instruction"],
    )


# SelectData is a subclass of EventData
def on_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("Instruction ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan - 1]
    return selected_plan


with gr.Blocks(
    title="长篇小说生成器", css="footer {visibility: hidden}", theme="default"
) as demo:
    gr.Markdown(
        """
    # 长篇小说生成器
    使用**RecurrentGPT**与**文心一言**生成任意篇幅长度的小说。
    """
    )
    with gr.Tab("自动生成"):
        with gr.Row():
            with gr.Column():
                with gr.Blocks():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            novel_type = gr.Textbox(label="类型", placeholder="如：玄幻")
                        with gr.Column(scale=2, min_width=400):
                            description = gr.Textbox(label="描述")
                btn_init = gr.Button("初始化", variant="primary")
                gr.Examples(
                    ["玄幻", "奇幻", "武侠", "仙侠", "都市", "现实", "历史", "军事", "游戏", "体育"],
                    inputs=[novel_type],
                )
                written_paras = gr.Textbox(label="已完成段落 (可编辑)", max_lines=21, lines=21)
            with gr.Column():
                with gr.Blocks():
                    gr.Markdown("### 记忆模块\n")
                    short_memory = gr.Textbox(label="短期记忆 (可编辑)", max_lines=3, lines=3)
                    long_memory = gr.Textbox(label="长期记忆 (可编辑)", max_lines=6, lines=6)
                with gr.Blocks():
                    gr.Markdown("### 指令模块\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="指令 1 (可编辑)", max_lines=4, lines=4
                        )
                        instruction2 = gr.Textbox(
                            label="指令 2 (可编辑)", max_lines=4, lines=4
                        )
                        instruction3 = gr.Textbox(
                            label="指令 3 (可编辑)", max_lines=4, lines=4
                        )
                    selected_plan = gr.Textbox(
                        label="修订的指令",
                        max_lines=2,
                        lines=2,
                    )

                btn_step = gr.Button("下一步", variant="primary")

        btn_init.click(
            init,
            inputs=[novel_type, description],
            outputs=[
                short_memory,
                long_memory,
                written_paras,
                instruction1,
                instruction2,
                instruction3,
            ],
        )
        btn_step.click(
            step,
            inputs=[
                short_memory,
                long_memory,
                instruction1,
                instruction2,
                instruction3,
                written_paras,
            ],
            outputs=[
                short_memory,
                long_memory,
                written_paras,
                selected_plan,
                instruction1,
                instruction2,
                instruction3,
            ],
        )

    with gr.Tab("人机互助 HITL"):
        with gr.Row():
            with gr.Column():
                with gr.Blocks():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            novel_type = gr.Textbox(label="类型", placeholder="玄幻")
                        with gr.Column(scale=2, min_width=400):
                            description = gr.Textbox(label="描述")
                btn_init = gr.Button("初始化", variant="primary")
                gr.Examples(
                    ["玄幻", "奇幻", "武侠", "仙侠", "都市", "现实", "历史", "军事", "游戏", "体育"],
                    inputs=[novel_type],
                )
                written_paras = gr.Textbox(label="已完成段落 (可编辑)", max_lines=23, lines=23)
            with gr.Column():
                with gr.Blocks():
                    gr.Markdown("### 记忆模块\n")
                    short_memory = gr.Textbox(label="短期记忆 (可编辑)", max_lines=3, lines=3)
                    long_memory = gr.Textbox(label="长期记忆 (可编辑)", max_lines=6, lines=6)
                with gr.Blocks():
                    gr.Markdown("### 指令 模块\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="指令 1",
                            max_lines=3,
                            lines=3,
                            interactive=False,
                        )
                        instruction2 = gr.Textbox(
                            label="指令 2",
                            max_lines=3,
                            lines=3,
                            interactive=False,
                        )
                        instruction3 = gr.Textbox(
                            label="指令 3",
                            max_lines=3,
                            lines=3,
                            interactive=False,
                        )
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            selected_plan = gr.Radio(
                                ["指令 1", "指令 2", "指令 3"],
                                label="指令选择",
                            )
                            #  info="Select the instruction you want to revise and use for the next step generation.")
                        with gr.Column(scale=3, min_width=300):
                            selected_instruction = gr.Textbox(
                                label="选择的指令 (可编辑)",
                                max_lines=5,
                                lines=5,
                            )

                btn_step = gr.Button("下一步", variant="primary")

        btn_init.click(
            init,
            inputs=[novel_type, description],
            outputs=[
                short_memory,
                long_memory,
                written_paras,
                instruction1,
                instruction2,
                instruction3,
            ],
        )
        btn_step.click(
            controled_step,
            inputs=[short_memory, long_memory, selected_instruction, written_paras],
            outputs=[
                short_memory,
                long_memory,
                written_paras,
                instruction1,
                instruction2,
                instruction3,
            ],
        )
        selected_plan.select(
            on_select,
            inputs=[instruction1, instruction2, instruction3],
            outputs=[selected_instruction],
        )

    demo.queue()

if __name__ == "__main__":
    demo.launch(server_port=8005, share=True, server_name="0.0.0.0", show_api=False)
