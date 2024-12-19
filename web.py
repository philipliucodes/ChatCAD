from glob import glob
import os
import time
import gradio as gr
from chat_bot import gpt_bot
import nibabel as nib
import cv2
from datetime import datetime

# os.environ["http_proxy"]="http://127.0.0.1.1:7890"
# os.environ["https_proxy"]="http://127.0.0.1:7890"

title = """<h1 align="center">ChatCAD Plus</h1>"""
description = """**This is an early testing version of ChatCAD-plus. Feedback and contributions are welcome.<br>-Upload images such as chest X-rays or dental X-rays to the chat box to receive ChatCAD-plus's analysis of the image.<br>-You can continue to interact with ChatCAD-plus to further understand possible conditions.<br>-ChatCAD-plus will provide relevant resource links when necessary.**"""
chatbot_bindings =  None
chatbot = None

def concat_history(message_history: list) -> str:
    ret = ""
    for event in message_history:
        ret += f"{event['role']}: {event['content']}\n"
    return ret

def chatcad(history, message_history):
    if chatbot_bindings is None:
        response = '''**Please enter the API key first, then click save.**'''
        history[-1][1] = response
        yield history
    else:
        ref_record = concat_history(message_history)
        user_message = history[-1][0]
        # Chatbot interaction starts here
        # response = '''**That's cool!**'''
        if isinstance(history[-1][0], str):
            prompt = history[-1][0]
            response = chatbot_bindings.chat(prompt, ref_record)
            message_history += [{"role": "user", "content": user_message}]
        else:
            # response, modality = chatbot_bindings.report_zh(history[-1][0]['name'])
            response, modality = chatbot_bindings.report_zh(history[-1][0][0])
            message_history[-1] = {"role": "user", "content": f"The user uploaded a {modality} and requested a diagnostic result."}

        history[-1][1] = response
        message_history += [{"role": "assistant", "content": response}]
        
        yield history, message_history

def add_text(history, text):
    history = history + [(text, None)]
    return history, None

def add_file(history, file):
    # This is file path
    print(file.name)
    img_path = file.name
    update_time = str(datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")
    if file.name.endswith(".nii.gz"):
        img = nib.load(img_path)
        _, _, queue = img.dataobj.shape
        temp_img = img.dataobj[:, :, queue//2].T
        cv2.imwrite("./imgs/temp/" + str(update_time) + ".jpg", temp_img)
        img_path = "./imgs/temp/" + str(update_time) + ".jpg"
        
    history = history + [((img_path,), None)]
    return history

def add_state(info, history, message_key, message_history):
    try:
        global chatbot_bindings
        chatbot_bindings = gpt_bot(engine="gpt-3.5-turbo", api_key=info)
        chatbot_bindings.start()
        # chatbot_bindings = 1
        response = '**Initialization successful!**'
    except Exception as e:
        chatbot_bindings = None
        response = '**Initialization failed. Please enter the correct OpenAI key.**'
        print(f"Error during chatbot initialization: {e}")  # Log the error for debugging
            
    message_key = [{"role": "api_key", "content": info}]
    message_history += [{"role": "system", "content": response}]
        
    history = history + [(None, response)]
    return history, message_key, message_history

def clean_data():
    return [{"role": "system", "content": description}], None, None

def example_img(i, history):
    return i

callback = gr.CSVLogger()

with gr.Blocks(css="""#col_container1 {margin-left: auto; margin-right: auto;}
                      #col_container2 {margin-left: auto; margin-right: auto;}
                      #chatbot {height: 770px;}
                      #upload_btn {height: auto;}""") as demo:
    gr.HTML(title)
    
    user_history = gr.State([])
    user_key = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=0.2):
            with gr.Row():
                #top_p, temperature, top_k, repetition_penalty
                with gr.Accordion("Settings", open=True):
                    with gr.Row():
                        api_key_input = gr.Textbox(placeholder="Please enter the API key.", label="API key")
                    with gr.Row():
                        api_key_submit = gr.Button("Save")
            with gr.Row():
                gr.Markdown("### Please upload the medical images you wish to consult. If you don't have any images, you can download the sample images below.")
            with gr.Row():
                upload_file = gr.UploadButton("📁 Upload Image", file_types=["file"], elem_id='upload_btn').style(size='lg')
            with gr.Row():
                img_i = gr.Image(show_label=False, type="numpy", interactive=False)
                gr.Examples(
                    [ 
                        os.path.join(os.path.dirname(__file__), "imgs/examples/chest.jpg"),
                        os.path.join(os.path.dirname(__file__), "imgs/examples/tooth.jpg"),
                    ],
                    img_i,
                    img_i,
                    example_img,
                    label="Sample Images"
                )
        with gr.Column(scale=0.8):
            with gr.Row():
                with gr.Column(elem_id = "col_container1"):
                    chatbot = gr.Chatbot(value=[(None, description)], label="ChatCAD Plus", elem_id='chatbot').style(height=700) #c
            with gr.Row():
                with gr.Column(elem_id = "col_container2", scale=0.85):
                    inputs = gr.Textbox(label="Chat Box", placeholder="Please enter text or upload an image.") #t
                with gr.Column(elem_id = "col_container2", scale=0.15, min_width=0):
                    with gr.Row():
                        inputs_submit = gr.Button("Send", elem_id='inputs_submit')
                    with gr.Row():
                        clean_btn = gr.Button("Clear", elem_id='clean_btn')
                    
        
    api_key_submit.click(add_state, [api_key_input, chatbot, user_key, user_history], [chatbot, user_key, user_history])
    
    inputs_submit.click(add_text, [chatbot, inputs], [chatbot, inputs]).then(
        chatcad, [chatbot, user_history], [chatbot, user_history]
    )
    
    clean_btn.click(clean_data, [], [chatbot, inputs, img_i])
    clean_btn.click(lambda: None, None, chatbot, queue=False).success(clean_data, [], [user_history, inputs, img_i])
    
    inputs.submit(add_text, [chatbot, inputs], [chatbot, inputs]).then(
        chatcad, [chatbot, user_history], [chatbot, user_history]
    )
    
    upload_file.upload(add_file, [chatbot, upload_file], [chatbot]).then(
        chatcad, [chatbot, user_history], [chatbot, user_history]
    )
    
    # 127.0.0.1.1:7890
    demo.queue().launch(server_port=4900, server_name="0.0.0.0", favicon_path="shtu.ico", share=True)
    # demo.queue().launch(server_port=4900, server_name="127.0.0.1", favicon_path="shtu.ico")
    # demo.queue().launch(server_port=4900, server_name="127.0.0.1", favicon_path="shtu.ico")