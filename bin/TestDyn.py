import gradio as gr


cnt = 0

def words(num):

    cnt = num
    print(num)
    update_show = []
    update_hide = []
    for i in range(cnt):
        text = gr.Label(str(i), visible=True)
        row = gr.Button(visible=True)
        update_show.append(text)
        update_show.append(row)
    for i in range(10 - cnt):
        text = gr.Label(str(i), visible=False)
        row = gr.Button(visible=False)
        update_hide.append(text)
        update_hide.append(row)
    print(update_show, update_hide, cnt)
    return update_show + update_hide

rows = []

with gr.Blocks() as demo:
    with gr.Column():
        for i in range(10):
            with gr.Row():
                text = gr.Label(str(i), visible=False)
                btn = gr.Button(visible=False)
            #row.add(btn)
            rows.append(text)
            rows.append(btn)

    dropdown = gr.Dropdown(choices = range(2,10))
    dropdown.change(words, dropdown, rows)

demo.launch()
