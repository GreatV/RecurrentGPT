#!/bin/bash
export QIANFAN_API_KEY=""
export QIANFAN_SECRET_KEY=""

iteration=10
outfile=response.txt
init_prompt=init_prompt.json
topic="凡人小伙修仙流"
type="玄幻"


options="\
        --iter $iteration\
        --r_file $outfile \
        --init_prompt $init_prompt \
        --topic $topic \
        --type $type \
        "
python main.py $options

# python gradio_server.py
