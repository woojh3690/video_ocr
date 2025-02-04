let video = document.getElementById('video');
let videoUpload = document.getElementById('video-upload');
let startOcrBtn = document.getElementById('start-ocr-btn');
let downloadBtn = document.getElementById('download-btn');
let ocrProgressContainer = document.getElementById('ocr-progress-container');

let boundingBox = document.getElementById('bounding-box');
let handles = document.querySelectorAll('.handle');

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

let vfilename = null;

// DOM 요소들: 작업 목록 뷰와 OCR 생성 뷰
const taskListView = document.getElementById('task-list-view');
const ocrCreationView = document.getElementById('ocr-creation-view');
const taskListTableBody = document.querySelector('#task-list tbody');

const newOcrBtn = document.getElementById('new-ocr-btn');
const backToListBtn = document.getElementById('back-to-list-btn');

// WebSocket 연결 (단일 연결)
let ws = new WebSocket(`ws://${location.host}/ws/tasks`);

ws.onopen = () => {
    console.log("WebSocket 연결 성공");
};

ws.onmessage = (event) => {
    let message = JSON.parse(event.data);
    // message는 { task_id, progress, status, estimated_completion, error, ... } 형태
    updateTaskRow(message);
};

ws.onerror = (err) => {
    console.error("WebSocket 오류:", err);
};

// 작업 목록 초기 로드
fetch('/tasks/')
    .then(response => response.json())
    .then(data => {
        // data는 tasks 딕셔너리 형태 { task_id: { ... }, ... }
        for (let taskId in data) {
            let task = data[taskId];
            task.task_id = taskId;
            updateTaskRow(task);
        }
    })
    .catch(err => console.error(err));

// --- 함수 정의 ---

// 작업 목록 테이블의 row 업데이트 (존재하면 수정, 없으면 생성)
function updateTaskRow(task) {
    let taskId = task.task_id;
    let existingRow = document.getElementById("task-row-" + taskId);
    let progress = task.progress || 0;
    let status = task.status || "";
    let estimated = (typeof task.estimated_completion !== "undefined") ? task.estimated_completion : "TDB";
    let videoFile = task.video_filename || "";
    
    // progress bar HTML
    let progressBarHtml = `<div class="progress">
        <div class="progress-bar" role="progressbar" style="width: ${progress}%;" aria-valuenow="${progress}" aria-valuemin="0" aria-valuemax="100"></div>
    </div>`;
    
    // 중지 or 삭제 버튼
    let btnHtml = "";
    if (status === "running") {
        btnHtml = `<button class="btn btn-warning btn-sm cancel-btn" data-task-id="${taskId}">중지</button>`;
    } else {
        btnHtml = `<button class="btn btn-danger btn-sm delete-btn" data-task-id="${taskId}">삭제</button>`;
    }
    
    let rowHtml = `
        <td>${taskId}</td>
        <td>${videoFile}</td>
        <td>${progressBarHtml}</td>
        <td>${status}</td>
        <td>${estimated}</td>
        <td>${btnHtml}</td>
    `;
    
    if (existingRow) {
        existingRow.innerHTML = rowHtml;
    } else {
        let tr = document.createElement("tr");
        tr.id = "task-row-" + taskId;
        tr.innerHTML = rowHtml;
        taskListTableBody.appendChild(tr);
    }
    
    // 중지 버튼 이벤트 연결
    document.querySelectorAll(".cancel-btn").forEach(btn => {
        btn.onclick = function() {
            let id = this.getAttribute("data-task-id");
            cancelTask(id);
        };
    });

    // 삭제 버튼 이벤트 연결
    document.querySelectorAll(".delete-btn").forEach(btn => {
        btn.onclick = function() {
            let id = this.getAttribute("data-task-id");
            deleteTask(id);
        };
    });
}

// CANCEL 요청을 보내고 작업 중지
function cancelTask(taskId) {
    let formData = new FormData();
    formData.append('task_id', taskId);
    fetch(`/cancel_ocr`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.detail);
    })
    .catch(err => console.error(err));
}

// DELETE 요청을 보내고 row 제거
function deleteTask(taskId) {
    fetch(`/tasks/${taskId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        // 삭제 성공하면 테이블 row 제거
        let row = document.getElementById("task-row-" + taskId);
        if (row) {
            row.remove();
        }
    })
    .catch(err => console.error(err));
}

// 뷰 전환 함수
function switchToOcrCreationView() {
    taskListView.style.display = 'none';
    ocrCreationView.style.display = 'block';
}

function switchToTaskListView() {
    ocrCreationView.style.display = 'none';
    taskListView.style.display = 'block';
}

// 이벤트: "새 OCR 작업 추가" 버튼 -> OCR 생성 뷰로 전환
newOcrBtn.addEventListener('click', function() {
    switchToOcrCreationView();
});

// 이벤트: "작업 큐로 돌아가기" 버튼 -> 작업 목록 뷰로 전환
backToListBtn.addEventListener('click', function() {
    switchToTaskListView();
});

// --- 기존 OCR 생성 뷰 관련 코드 (수정) ---

// 비디오 업로드 및 로드
videoUpload.addEventListener('change', function(e) {
    let file = e.target.files[0];
    if (file) {
        // 비디오 업로드 및 로드
        let formData = new FormData();
        formData.append('file', file);
        fetch('/upload_video/', {
            method: 'POST',
            body: formData
        }).then(response => response.json())
        .then(data => {
            vfilename = data.filename;
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
                boundingBox.style.height = (video.clientHeight - video.controls.offsetHeight) + 'px';
            };
        });
    }
});

// 드래그 핸들 이벤트 처리
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
    intervalValue = parseFloat(event.target.value);
});

// OCR 시작: 이제 POST /start_ocr/를 호출하면 task_id를 받고, WebSocket 업데이트로 진행률이 표시됨.
startOcrBtn.addEventListener('click', async function() {
    // 초기화
    downloadBtn.style.display = 'none';
    
    // 바운딩 박스의 위치와 크기 계산
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
    formData.append('video_filename', videoUpload.files[0].name);
    formData.append('x', x);
    formData.append('y', y);
    formData.append('width', width);
    formData.append('height', height);
    formData.append('interval_value', intervalValue);
    
    try {
        let response = await fetch('/start_ocr/', {
            method: 'POST',
            body: formData
        });
        let data = await response.json();
        switchToTaskListView();
    } catch (err) {
        alert('OCR 작업 시작 중 오류 발생: ' + err.message);
    }
});

// SRT 다운로드 버튼 이벤트
downloadBtn.addEventListener('click', function() {
    // vfilename가 현재 작업의 비디오 파일명임.
    window.location.href = `/download_srt/${vfilename}`;
});
