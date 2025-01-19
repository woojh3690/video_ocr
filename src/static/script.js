let video = document.getElementById('video');
let uploadBtn = document.getElementById('upload-btn');
let startOcrBtn = document.getElementById('start-ocr-btn');
let downloadBtn = document.getElementById('download-btn');
let progressContainer = document.getElementById('progress-container');
let progressBar = document.getElementById('progress');

let boundingBox = document.getElementById('bounding-box');
let handles = document.querySelectorAll('.handle');

let videoFile;
let intervalValue = 0.3;
let dragging = false;
let dragDirection = '';
let startY = 0;
let startX = 0;
let startHeight = 0;
let startWidth = 0;
let videoWidth = 0;
let videoHeight = 0;
let startTop = 0;
let startLeft = 0;

vfilename = null
document.getElementById('video-upload').addEventListener('change', function(e) {
    videoFile = e.target.files[0];
    if (videoFile) {
        // 비디오 업로드 및 로드
        let formData = new FormData();
        formData.append('file', videoFile);
        fetch('/upload_video/', {
            method: 'POST',
            body: formData
        }).then(response => response.json())
        .then(data => {
            // 비디오를 서버에서 불러오기
            vfilename = data.filename
            let url = `/videos/${data.filename}`;
            video.src = url;
            video.load();
            video.onloadedmetadata = function() {
                videoWidth = video.videoWidth;
                videoHeight = video.videoHeight;

                // 비디오 컨테이너 크기 조정
                let videoContainer = document.getElementById('video-container');
                videoContainer.style.width = video.clientWidth + 'px';
                videoContainer.style.height = video.clientHeight + 'px';

                // 바운딩 박스 초기화
                boundingBox.style.top = '0px';
                boundingBox.style.left = '0px';
                boundingBox.style.width = video.clientWidth + 'px';
                boundingBox.style.height = video.clientHeight - video.controls.offsetHeight + 'px'; // 비디오 컨트롤 높이 제외
            };
        });
    }
});

// 드래그 핸들에 이벤트 리스너 추가
handles.forEach(function(handle) {
    handle.addEventListener('mousedown', function(e) {
        dragging = true;
        dragDirection = e.target.classList.contains('top') ? 'top' :
                        e.target.classList.contains('bottom') ? 'bottom' :
                        e.target.classList.contains('left') ? 'left' :
                        e.target.classList.contains('right') ? 'right' : '';
        startY = e.clientY;
        startX = e.clientX;
        startHeight = boundingBox.offsetHeight;
        startWidth = boundingBox.offsetWidth;
        startTop = boundingBox.offsetTop;
        startLeft = boundingBox.offsetLeft;
        e.preventDefault();
    });
});

document.addEventListener('mousemove', function(e) {
    if (dragging) {
        let dy = e.clientY - startY;
        let dx = e.clientX - startX;

        if (dragDirection === 'top') {
            let newHeight = startHeight - dy;
            let newTop = startTop + dy;
            if (newHeight > 20 && newTop >= 0) {
                boundingBox.style.height = newHeight + 'px';
                boundingBox.style.top = newTop + 'px';
            } else if (newTop < 0) {
                boundingBox.style.height = startHeight + startTop + 'px';
                boundingBox.style.top = '0px';
            }
        } else if (dragDirection === 'bottom') {
            let newHeight = startHeight + dy;
            let maxHeight = video.clientHeight - startTop;
            if (newHeight > 20 && newHeight <= maxHeight) {
                boundingBox.style.height = newHeight + 'px';
            } else if (newHeight > maxHeight) {
                boundingBox.style.height = maxHeight + 'px';
            }
        } else if (dragDirection === 'left') {
            let newWidth = startWidth - dx;
            let newLeft = startLeft + dx;
            if (newWidth > 20 && newLeft >= 0) {
                boundingBox.style.width = newWidth + 'px';
                boundingBox.style.left = newLeft + 'px';
            } else if (newLeft < 0) {
                boundingBox.style.width = startWidth + startLeft + 'px';
                boundingBox.style.left = '0px';
            }
        } else if (dragDirection === 'right') {
            let newWidth = startWidth + dx;
            let maxWidth = video.clientWidth - startLeft;
            if (newWidth > 20 && newWidth <= maxWidth) {
                boundingBox.style.width = newWidth + 'px';
            } else if (newWidth > maxWidth) {
                boundingBox.style.width = maxWidth + 'px';
            }
        }
        e.preventDefault();
    }
});

document.addEventListener('mouseup', function(e) {
    dragging = false;
    dragDirection = '';
});

// interval 설정
document.getElementById("intervalInput").addEventListener("input", (event) => {
    const inputValue = parseFloat(event.target.value);
    intervalValue = inputValue;
});

// OCR 시작
startOcrBtn.addEventListener('click', async function() {
    // 진행 바 및 예상 완료 시간 초기화
    progressBar.value = 0;
    let estimatedCompletion = document.getElementById('estimated-completion');
    if (estimatedCompletion) {
        estimatedCompletion.textContent = '';
    }
    downloadBtn.style.display = 'none';

    // 바운딩 박스의 위치와 크기를 계산
    let videoRect = video.getBoundingClientRect();
    let boxRect = boundingBox.getBoundingClientRect();

    let x = boxRect.left - videoRect.left;
    let y = boxRect.top - videoRect.top;
    let width = boxRect.width;
    let height = boxRect.height;

    // 비율 조정
    let scaleX = video.videoWidth / video.clientWidth;
    let scaleY = video.videoHeight / video.clientHeight;

    x = Math.round(x * scaleX);
    y = Math.round(y * scaleY);
    width = Math.round(width * scaleX);
    height = Math.round(height * scaleY);

    let formData = new FormData();
    formData.append('video_filename', videoFile.name);
    formData.append('x', x);
    formData.append('y', y);
    formData.append('width', width);
    formData.append('height', height);
    formData.append('interval_value', intervalValue);

    try {
        // 작업 시작 시간 기록
        let startTime = Date.now();

        // 서버로 OCR 작업 요청을 전송합니다.
        const response = await fetch('/start_ocr/', {
            method: 'POST',
            body: formData
        });

        if (!response.body) {
            throw new Error('No response body received.');
        }

        // 진행 바 및 다운로드 버튼 표시
        progressContainer.style.display = 'block';

        const reader = response.body.getReader();
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // 서버에서 받은 진행 상태 데이터를 처리합니다.
            let lines = new TextDecoder().decode(value);
            lines = lines.split('\n').filter(line => line.trim() !== '');
            let progressText = buffer = lines[lines.length - 1];
            progressText = progressText.substring('data:'.length).trim();
            const progressData = JSON.parse(progressText);
            progressBar.value = progressData.progress;

            // 경과 시간 및 남은 예상 시간 계산
            let currentTime = Date.now();
            let elapsedTime = (currentTime - startTime) / 1000; // 초 단위 경과 시간

            let estimatedCompletion = document.getElementById('estimated-completion');
            if (progressData.progress > 0 && progressData.progress < 100) {
                let remainingTime = (elapsedTime * (100 - progressData.progress)) / progressData.progress;
                // 남은 예상 시간 표시
                estimatedCompletion.textContent = '남은 예상 시간: 약 ' + formatTime(remainingTime);
            } else if (progressData.progress === 100) {
                downloadBtn.style.display = 'block';
                // 남은 예상 시간 제거
                estimatedCompletion.textContent = 'OCR 완료';
            }
        }
    } catch (err) {
        alert('OCR 처리 중 오류가 발생했습니다: ' + err.message);
    }
});

function formatTime(seconds) {
    // 초를 시간, 분, 초로 변환
    let hrs = Math.floor(seconds / 3600);
    let mins = Math.floor((seconds % 3600) / 60);
    let secs = Math.floor(seconds % 60);

    // 각 부분을 2자리 숫자로 변환 (ex: 5 -> "05")
    let formattedHrs = hrs > 0 ? String(hrs).padStart(2, '0') + ':' : '';
    let formattedMins = String(mins).padStart(2, '0');
    let formattedSecs = String(secs).padStart(2, '0');

    // 시간 형식 반환
    return formattedHrs + formattedMins + ':' + formattedSecs;
}

downloadBtn.addEventListener('click', function() {
    window.location.href = `/download_srt/${vfilename}`;
});
