import os
import sys
import base64
import json
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler

system_prompt = 'You are a ocr assistant. Extract all the subtitles and text in JSON format. \
Group the subtitles according to color of the subtitles; \
subtitles with the same color belong to the same group.\n\
Use the following JSON format: {\"ocr_subtitles_group\":[[\"group1 first subtitle\",\"group1 second subtitle\",\"...\"],\
[\"group2 first subtitle\",\"group2 second subtitle\",\"...\"]]}\n\
If there is no subtitles then: {\"ocr_subtitles_group\":[]}'


class SuppressStdoutStderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )

        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )

        self.outnull_file.close()
        self.errnull_file.close()

with SuppressStdoutStderr():
    chat_handler = MiniCPMv26ChatHandler(clip_model_path="models/mmproj-model-f16.gguf")
    llm = Llama(
        model_path="models/ggml-model-Q8_0.gguf",
        chat_handler=chat_handler,
        n_ctx=16384,
        n_gpu_layers=-1,
        verbose=False,
    )

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/jpg;base64,{base64_data}"

def process_images(folder_path):
    file_list = os.listdir(folder_path)

    # 폴더 내의 모든 파일을 순회
    for filename in sorted(file_list):
        if ".533" in filename or "842" in filename:
            continue
        file_path = os.path.join(folder_path, filename)
        data_uri = image_to_base64_data_uri(file_path)

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user", 
                "content": [{"type": "image_url", "image_url": {"url": data_uri}}]
            },
        ]

        response_format = {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "ocr_subtitles_group": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    }
                },
                "required": [
                    "ocr_subtitles_group"
                ]
            },
        }

        with SuppressStdoutStderr():
            response = llm.create_chat_completion(
                messages=messages,
                response_format=response_format,
                temperature=0.0,
            )

        content = response["choices"][0]["message"]["content"]
        ocr_subtitles_group = json.loads(content)["ocr_subtitles_group"]
        print(ocr_subtitles_group)


def normalize_to_nested_list(json_obj):
    # 단일 문자열을 [["text"]]로 변환
    if isinstance(json_obj, str):
        return [[json_obj]]

    # 리스트인지 확인
    if isinstance(json_obj, list):
        # 리스트 안에 중첩 리스트가 없으면 [["..."]]로 변환
        if not any(isinstance(i, list) for i in json_obj):
            return [json_obj]

    # 이미 올바른 형식이면 그대로 반환
    return json_obj

if __name__ == "__main__":
    folder_path = './test_ocr/test_image'  # 이미지 폴더 경로를 설정하세요
    process_images(folder_path)
