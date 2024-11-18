import os
import json
import ollama

system_prompt = 'Extract all the subtitles and text in JSON format. \
Group the subtitles according to color of the subtitles; \
subtitles with the same color belong to the same group. \
Use the following JSON format: \n\n\
{\"ocr_subtitles_group\":[[\"group1 first subtitle\",\"group1 second subtitle\"],[\"group2 first subtitle\",\"group2 second subtitle\"]}\
\n\nIf there is no subtitles then:\
{\"ocr_subtitles_group\":[[]]}'

ollama_url = os.environ["OLLAMA_URL"]
client = ollama.Client(ollama_url)

def process_images(folder_path):
    file_list = os.listdir(folder_path)

    # 폴더 내의 모든 파일을 순회
    for filename in sorted(file_list):
        response = client.chat(
            model='llama3.2-vision:90b',
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'images': [os.path.join(folder_path, filename)]
                }
            ],
            format="json",
            stream=False,
            options={
                'temperature': 0,
                'num_predict': 512
            }
        )
        content = response['message']['content']
        ocr_subtitles_group = json.loads(content)["ocr_subtitles_group"]
        ocr_subtitles_group = normalize_to_nested_list(ocr_subtitles_group)
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
    folder_path = './test_image'  # 이미지 폴더 경로를 설정하세요
    process_images(folder_path)
