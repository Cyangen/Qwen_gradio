from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import gradio as gr
from timeit import default_timer as timer
# from PIL import Image
import torch, os

models_dir =os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
# default: Load the model on the available device(s)
model_name = "Qwen2-VL-7B-Instruct"
# model_name = "Qwen2-VL-7B-Instruct-AWQ"
# model_name = "Qwen2-VL-7B-Instruct-GPTQ-Int4"
model_path = os.path.join(models_dir, model_name)


model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map=0 #changed device_map from "auto" and torch_dtype from "auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
# processor = AutoProcessor.from_pretrained(model_path)

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)



def qwen_predict(question, image):
    start = timer()
    #Data to send to model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text", 
                    "text": question
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_list = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    #Conversion of model output(list) to structured str
    output_text = ""
    for segment in output_list:
        output_text += segment
    time_taken = f"{timer()-start:.2f}s"

    print(f"Done in: {time_taken}")
    return output_text, time_taken

#Gradio interface
interface = gr.Interface(
    title= "Qwen2-VL-7B",
    fn=qwen_predict,
    inputs=[gr.Textbox(label="Question", placeholder="Enter your question here", lines=1, max_lines=10), gr.Image(type="pil")], #Conversion to Pillow format for base64 encoding
    outputs=[gr.Textbox(label="Answer", lines=1, max_lines=10), gr.Textbox(label="Time Taken", lines=1)],
)
interface.launch(share=True)