import requests
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
import re
from PIL import Image

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe the symptoms displayed in this image. Only describe the image medically. Do not list treatment."}
    ]}
]

def process(image):
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=1000)
    text = processor.decode(output[0])
    text = re.sub(r".*<\|end_header_id\|>", '', text)
    text = re.sub(r"<|eot_id|>", '', text)
    return text

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create a Description of an image')
    parser.add_argument('--imgpath', metavar='path', required=False,
                        help='the path to the desired image')
    args = parser.parse_args()
    img_path = "child_rash.jpg"
    if args.imgpath:
        img_path = args.imgpath
    img = Image.open(img_path)
    print(process(img))
