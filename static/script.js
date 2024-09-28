let video = document.getElementById('video');
let uploadBtn = document.getElementById('upload-btn');
let startOcrBtn = document.getElementById('start-ocr-btn');
let downloadBtn = document.getElementById('download-btn');
let progressContainer = document.getElementById('progress-container');
let progressBar = document.getElementById('progress');

let boundingBox = document.getElementById('bounding-box');
let handles = document.querySelectorAll('.handle');

let videoFile;
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
            // alert 제거
            // 비디오를 서버에서 불러오기
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

startOcrBtn.addEventListener('click', function() {
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

    fetch('/start_ocr/', {
        method: 'POST',
        body: formData
    }).then(response => response.json())
    .then(data => {
        progressContainer.style.display = 'block';
        checkProgress();
    });
});

function checkProgress() {
    fetch('/progress/')
    .then(response => response.json())
    .then(data => {
        progressBar.value = data.progress;
        if (data.progress < 100) {
            setTimeout(checkProgress, 1000);
        } else {
            alert('OCR 완료');
            downloadBtn.style.display = 'block';
        }
    });
}

downloadBtn.addEventListener('click', function() {
    window.location.href = '/download_srt/';
});
