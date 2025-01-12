import os
import copy
import json

from pydantic import BaseModel, ValidationError
import ollama

system_prompt = 'OCR all the text from image following JSON: \n\
{\"texts\":\"example\"}'

class OcrSubtitleGroup(BaseModel):
    texts: list[str]

def make_few_shot_template(folder_path):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    example_file = os.path.join(folder_path, "answer.txt")
    with open(example_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue

            example_image = os.path.join(folder_path, f"shot{i}.jpg")
            messages.append({
                "role": "user",
                "images": [example_image]
            })
            messages.append({
                "role": "assistant",
                "content": line
            })
    return messages

async def process_images(host_url, test_image_path, few_shot_path):
    file_list = os.listdir(test_image_path)
    client = ollama.AsyncClient(host=host_url)

    messages_template = make_few_shot_template(few_shot_path)

    # 폴더 내의 모든 파일을 순회
    for filename in sorted(file_list):
        file_path = os.path.join(test_image_path, filename)

        messages = copy.deepcopy(messages_template)
        messages.append(
            {
                "role": "user",
                "images": [file_path]
            }
        )

        response = await client.chat(
            model='minicpm-v:latest',
            format=OcrSubtitleGroup.model_json_schema(),
            options={
                'temperature': 0,
                'num_predict': 512
            },
            messages=messages,
        )

        content = response.message.content
        try:
            ocr_text = OcrSubtitleGroup.model_validate_json(content).texts
            print(f"{filename} : {ocr_text}")
        except ValidationError:
            print(f"{filename} 파일의 JSON 디코딩 에러:")
            print(content)
            continue

if __name__ == "__main__":
    import asyncio
    host_url = "http://localhost:11434"
    test_image_path = "./test/test_image"
    few_shot_path = "./src/few_shot"
    asyncio.run(process_images(host_url, test_image_path, few_shot_path))
