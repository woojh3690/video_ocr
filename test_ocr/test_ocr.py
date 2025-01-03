import os
import base64
import json
from pydantic import BaseModel, ValidationError
import ollama

system_prompt = 'Extract all the text in JSON format. \
Use the following JSON format: {\"text_in_image\":[\"text1\",\"text2\",\"...\",\"textN\"]}\n\
If there is no subtitles then: {\"text_in_image\":[]}'

class OcrSubtitleGroup(BaseModel):
    text_in_image: list[str]

async def process_images(folder_path):
    file_list = os.listdir(folder_path)
    client = ollama.AsyncClient(host="http://dev.iwaz.co.kr:9236")

    # 폴더 내의 모든 파일을 순회
    for filename in sorted(file_list):
        file_path = os.path.join(folder_path, filename)

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "images": [file_path]
            },
        ]

        response = await client.chat(
            model='llama3.2-vision:90b',
            format=OcrSubtitleGroup.model_json_schema(),
            options={
                'temperature': 0,
                'num_predict': 512
            },
            messages=messages,
        )

        content = response.message.content
        try:
            ocr_subtitles = OcrSubtitleGroup.model_validate_json(content).text_in_image
        except ValidationError:
            print(f"{filename} 파일의 JSON 디코딩 에러:")
            print(content)
            continue

        # 후처리
        ocr_text = "\n".join(ocr_subtitles)
        ocr_text = ocr_text.replace('\n', '\\n')

        print(f"{filename} : {ocr_text}")

if __name__ == "__main__":
    import asyncio
    folder_path = './test_ocr/test_image'  # 이미지 폴더 경로를 설정하세요
    asyncio.run(process_images(folder_path))
