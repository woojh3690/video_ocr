import docker
from docker.errors import NotFound, APIError

class DockerManager:
    def __init__(self, base_url: str):
        self.client = docker.DockerClient(base_url=base_url)

    def stop_container(self, name: str, timeout: int = 30) -> None:
        try:
            container = self.client.containers.get(name)
            container.stop(timeout=timeout)
            print(f"✅  컨테이너 '{name}' 중지 완료")
        except NotFound:
            print(f"❌  컨테이너 '{name}' 를 찾을 수 없습니다")
            raise
        except APIError as e:
            print(f"❌  Docker API 오류: {e.explanation}")
            raise

    def start_container(self, name: str) -> None:
        try:
            container = self.client.containers.get(name)
            container.start()
            print(f"✅  컨테이너 '{name}' 시작 완료")
        except NotFound:
            print(f"❌  컨테이너 '{name}' 를 찾을 수 없습니다")
            raise
        except APIError as e:
            print(f"❌  Docker API 오류: {e.explanation}")
            raise
