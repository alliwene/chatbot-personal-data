import gradio as gr  # type: ignore

from backend import load_db  # type: ignore


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    
    file = gr.File(type="file", label="Upload a PDF", file_types=[".pdf"])
    text = gr.Textbox(label="Question")

    btn = gr.Button(value="Submit")    
    clear = gr.ClearButton([file, btn, chatbot])

    btn.click(fn=load_db, inputs=[file, text], outputs=chatbot)

if __name__ == "__main__":
    demo.launch()

