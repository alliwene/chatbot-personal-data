import gradio as gr

from backend import load_db

# def similarity_search(pdf_file):
#     result = load_db(pdf_file)

#     return result


iface = gr.Interface(
    fn=load_db,
    inputs=gr.File(type="file", label="Upload a PDF", file_types=[".pdf"]),
    outputs=[gr.Textbox(label="similarity search results")],
) 

iface.launch(server_port=7862)