let video = document.getElementById('video');
let startOcrBtn = document.getElementById('start-ocr-btn');
let ocrProgressContainer = document.getElementById('ocr-progress-container');

// 현재 실행 중인 작업이 있는지 여부
let isTaskRunning = false;
// 각 작업의 상태 기록
const taskStatusMap = {};

const boundingBoxes = [
    { element: document.getElementById('bounding-box-1'), enabled: true },
    { element: document.getElementById('bounding-box-2'), enabled: false },
];
boundingBoxes.forEach((box) => {
    box.handles = box.element.querySelectorAll('.handle');
});
const secondRegionToggle = document.getElementById('toggle-second-region');

const dragState = {
    active: false,
    direction: '',
    startY: 0,
    startX: 0,
    startHeight: 0,
    startWidth: 0,
    startTop: 0,
    startLeft: 0,
    targetBox: null,
};
let videoWidth = 0;
let videoHeight = 0;

let vfilename = null;
let fileBrowser = document.getElementById('file-browser');
let fileList = document.getElementById('file-list');
let currentPathElem = document.getElementById('current-path');
let currentPath = '';

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

function encodePath(path) {
    return path.split('/').map(encodeURIComponent).join('/');
}

function loadDirectory(path = '') {
    fetch(`/browse/?path=${encodeURIComponent(path)}`)
        .then(resp => resp.json())
        .then(data => {
            currentPath = data.path;
            currentPathElem.textContent = '/' + (currentPath ? currentPath : '');
            fileList.innerHTML = '';
            if (currentPath) {
                const up = document.createElement('li');
                up.className = 'list-group-item list-group-item-action browser-item';
                const upIcon = document.createElement('span');
                upIcon.className = 'item-icon';
                upIcon.textContent = '\u2191';
                const upLabel = document.createElement('span');
                upLabel.className = 'item-label';
                upLabel.textContent = '..';
                up.appendChild(upIcon);
                up.appendChild(upLabel);
                up.onclick = () => {
                    const parts = currentPath.split('/');
                    parts.pop();
                    loadDirectory(parts.join('/'));
                };
                fileList.appendChild(up);
            }
            data.entries.forEach(entry => {
                const item = document.createElement('li');
                item.className = 'list-group-item list-group-item-action browser-item';
                const icon = document.createElement('span');
                icon.className = 'item-icon';
                icon.textContent = entry.is_dir ? '\uD83D\uDCC1' : '\uD83D\uDCC4';
                const label = document.createElement('span');
                label.className = 'item-label';
                label.textContent = entry.name + (entry.is_dir ? '/' : '');
                item.appendChild(icon);
                item.appendChild(label);
                if (entry.is_dir) {
                    item.onclick = () => {
                        loadDirectory(currentPath ? `${currentPath}/${entry.name}` : entry.name);
                    };
                } else {
                    item.onclick = () => {
                        selectVideo(currentPath ? `${currentPath}/${entry.name}` : entry.name);
                    };
                }
                fileList.appendChild(item);
            });
        });
}

function setBoundingBoxEnabled(box, enabled) {
    box.enabled = enabled;
    if (enabled) {
        box.element.classList.remove('hidden');
    } else {
        box.element.classList.add('hidden');
    }
    box.element.dataset.enabled = enabled ? 'true' : 'false';
}

function layoutDefaultRegions() {
    if (!video.videoWidth || !video.videoHeight) {
        return;
    }
    const videoContainer = document.getElementById('video-container');
    videoContainer.style.width = video.clientWidth + 'px';
    videoContainer.style.height = video.clientHeight + 'px';
    const usableHeight = Math.max(20, video.clientHeight);
    const halfWidth = Math.max(40, Math.floor(video.clientWidth / 2));

    const primaryWidth = secondRegionToggle.checked ? halfWidth - 4 : video.clientWidth;
    const primaryBox = boundingBoxes[0].element;
    primaryBox.style.top = '0px';
    primaryBox.style.left = '0px';
    primaryBox.style.width = `${primaryWidth}px`;
    primaryBox.style.height = `${usableHeight}px`;

    const secondaryBox = boundingBoxes[1].element;
    if (secondRegionToggle.checked) {
        const secondaryWidth = halfWidth - 4;
        secondaryBox.style.top = '0px';
        secondaryBox.style.height = `${usableHeight}px`;
        secondaryBox.style.width = `${secondaryWidth}px`;
        secondaryBox.style.left = `${Math.max(0, video.clientWidth - secondaryWidth)}px`;
    }
}

function attachHandleListeners() {
    boundingBoxes.forEach((box) => {
        box.handles.forEach((handle) => {
            handle.addEventListener('mousedown', function(e) {
                dragState.active = true;
                dragState.targetBox = box.element;
                dragState.direction = e.target.classList.contains('top') ? 'top' :
                    e.target.classList.contains('bottom') ? 'bottom' :
                    e.target.classList.contains('left') ? 'left' :
                    e.target.classList.contains('right') ? 'right' : '';
                dragState.startY = e.clientY;
                dragState.startX = e.clientX;
                dragState.startHeight = box.element.offsetHeight;
                dragState.startWidth = box.element.offsetWidth;
                dragState.startTop = box.element.offsetTop;
                dragState.startLeft = box.element.offsetLeft;
                e.preventDefault();
            });
        });
    });
}

function getEnabledRegions() {
    const regions = [];
    const videoRect = video.getBoundingClientRect();
    const scaleX = video.videoWidth / video.clientWidth;
    const scaleY = video.videoHeight / video.clientHeight;

    boundingBoxes.forEach((box, idx) => {
        if (!box.enabled || box.element.classList.contains('hidden')) {
            return;
        }
        const boxRect = box.element.getBoundingClientRect();
        let x = boxRect.left - videoRect.left;
        let y = boxRect.top - videoRect.top;
        let width = boxRect.width;
        let height = boxRect.height;
        x = Math.round(x * scaleX);
        y = Math.round(y * scaleY);
        width = Math.round(width * scaleX);
        height = Math.round(height * scaleY);
        regions.push({
            x,
            y,
            width,
            height,
            label: idx === 0 ? "region_1" : "region_2"
        });
    });
    return regions;
}

setBoundingBoxEnabled(boundingBoxes[0], true);
setBoundingBoxEnabled(boundingBoxes[1], false);
function selectVideo(path) {
    const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.mpg', '.mpeg', '.wmv'];
    const lower = path.toLowerCase();
    const isVideo = videoExtensions.some(ext => lower.endsWith(ext));
    if (!isVideo) {
        alert('비디오 파일을 선택해주세요.');
        return;
    }
    vfilename = path;
    const targetDiv = document.querySelector('#video-container');
    targetDiv.style.display = 'block';
    setBoundingBoxEnabled(boundingBoxes[0], true);
    setBoundingBoxEnabled(boundingBoxes[1], secondRegionToggle.checked);
    let url = `/videos/${encodePath(path)}`;
    video.src = url;
    video.load();
    video.onloadedmetadata = function() {
        videoWidth = video.videoWidth;
        videoHeight = video.videoHeight;
        layoutDefaultRegions();
    };
}

function updateRunningStatus(taskId, status) {
    if (typeof status === 'undefined') {
        delete taskStatusMap[taskId];
    } else {
        taskStatusMap[taskId] = status;
    }
    isTaskRunning = Object.values(taskStatusMap).some(s => s === 'running');
}

// 작업 제어 버튼을 상태에 맞게 설정
function setActionButtons(row, status, taskId) {
    const cell = row.querySelector('.action-cell');
    cell.innerHTML = '';

    if (status === 'running' || status === 'waiting') {
        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'btn btn-warning btn-sm cancel-btn';
        cancelBtn.innerText = '중지';
        cancelBtn.onclick = function() {
            cancelTask(taskId);
        };
        cell.appendChild(cancelBtn);
    } else if (status === 'cancelled' || status === 'cancelling' || status === 'error') {
        const resumeBtn = document.createElement('button');
        resumeBtn.className = 'btn btn-primary btn-sm resume-btn mr-1';
        resumeBtn.innerText = '재개';
        resumeBtn.onclick = function() {
            resumeTask(taskId);
        };

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn btn-danger btn-sm delete-btn';
        deleteBtn.innerText = '삭제';
        deleteBtn.onclick = function() {
            deleteTask(taskId);
        };

        cell.appendChild(resumeBtn);
        cell.appendChild(deleteBtn);
    } else {
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn btn-danger btn-sm delete-btn';
        deleteBtn.innerText = '삭제';
        deleteBtn.onclick = function() {
            deleteTask(taskId);
        };
        cell.appendChild(deleteBtn);
    }
}

// 작업 목록 테이블의 row 업데이트 (존재하면 수정, 없으면 생성)
function updateTaskRow(task) {
    let taskId = task.task_id;
    let progress = task.progress || 0;
    let status = task.status || "";
    let statusHtml;
    if (status === 'completed') {
        statusHtml = 'completed';
    } else {
        statusHtml = (status === 'waiting') ? '대기' : status;
        if (task.error) {
            statusHtml += `: ${task.error}`;
        }
    }
    let estimated = (typeof task.estimated_completion !== "undefined") ? task.estimated_completion : "TBD";
    let videoFile = task.video_filename || "";

    // 기존 row 검색
    let row = document.getElementById("task-row-" + taskId);

    if (!row) {
        // row가 없으면 새로 생성
        row = document.createElement("tr");
        row.id = "task-row-" + taskId;
        row.innerHTML = `
            <td class="video-file"></td>
            <td class="progress-cell">
                <div class="progress">
                    <div class="progress-bar" role="progressbar" style="width: ${progress}%;" aria-valuenow="${progress}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
            </td>
            <td class="status-cell">${statusHtml}</td>
            <td class="estimated-cell">${estimated}</td>
            <td class="action-cell"></td>
        `;
        const videoCell = row.querySelector('.video-file');
        videoCell.textContent = videoFile;
        videoCell.title = videoFile;
        taskListTableBody.appendChild(row);
        setActionButtons(row, status, taskId);
        updateRunningStatus(taskId, status);
    } else {
        // 기존 row가 있으면 변경된 부분만 업데이트
        const videoCell = row.querySelector('.video-file');
        videoCell.textContent = videoFile;
        videoCell.title = videoFile;

        let progressBar = row.querySelector('.progress-bar');
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);

        row.querySelector('.status-cell').innerHTML = statusHtml;
        row.querySelector('.estimated-cell').innerText = estimated;

        // 버튼 업데이트
        setActionButtons(row, status, taskId);
        updateRunningStatus(taskId, status);
    }
}

// CANCEL 요청을 보내고 작업 중지
function cancelTask(taskId) {
    let formData = new FormData();
    formData.append('task_id', taskId);
    fetch(`/cancel_ocr/`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.detail);
    })
    .catch(err => console.error(err));
}

// RESUME 요청을 보내고 작업 재시작
function resumeTask(taskId) {
    let formData = new FormData();
    formData.append('task_id', taskId);
    fetch(`/resume_ocr/`, {
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
        delete taskStatusMap[taskId];
        updateRunningStatus(taskId, undefined);
    })
    .catch(err => console.error(err));
}

// 뷰 전환 함수
function switchToOcrCreationView() {
    taskListView.style.display = 'none';
    ocrCreationView.style.display = 'block';
    loadDirectory('');
    const targetDiv = document.querySelector('#video-container');
    targetDiv.style.display = 'none';
    vfilename = null;
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

secondRegionToggle.addEventListener('change', function() {
    const enabled = secondRegionToggle.checked;
    setBoundingBoxEnabled(boundingBoxes[1], enabled);
    layoutDefaultRegions();
});

attachHandleListeners();

document.addEventListener('mousemove', function(e) {
    if (!dragState.active || !dragState.targetBox) {
        return;
    }
    const targetBox = dragState.targetBox;
    if (targetBox.classList.contains('hidden')) {
        return;
    }

    const dy = e.clientY - dragState.startY;
    const dx = e.clientX - dragState.startX;

    if (dragState.direction === 'top') {
        let newHeight = dragState.startHeight - dy;
        let newTop = dragState.startTop + dy;
        if (newHeight > 20 && newTop >= 0) {
            targetBox.style.height = newHeight + 'px';
            targetBox.style.top = newTop + 'px';
        } else if (newTop < 0) {
            targetBox.style.height = dragState.startHeight + dragState.startTop + 'px';
            targetBox.style.top = '0px';
        }
    } else if (dragState.direction === 'bottom') {
        let newHeight = dragState.startHeight + dy;
        let maxHeight = video.clientHeight - dragState.startTop;
        if (newHeight > 20 && newHeight <= maxHeight) {
            targetBox.style.height = newHeight + 'px';
        } else if (newHeight > maxHeight) {
            targetBox.style.height = maxHeight + 'px';
        }
    } else if (dragState.direction === 'left') {
        let newWidth = dragState.startWidth - dx;
        let newLeft = dragState.startLeft + dx;
        if (newWidth > 20 && newLeft >= 0) {
            targetBox.style.width = newWidth + 'px';
            targetBox.style.left = newLeft + 'px';
        } else if (newLeft < 0) {
            targetBox.style.width = dragState.startWidth + dragState.startLeft + 'px';
            targetBox.style.left = '0px';
        }
    } else if (dragState.direction === 'right') {
        let newWidth = dragState.startWidth + dx;
        let maxWidth = video.clientWidth - dragState.startLeft;
        if (newWidth > 20 && newWidth <= maxWidth) {
            targetBox.style.width = newWidth + 'px';
        } else if (newWidth > maxWidth) {
            targetBox.style.width = maxWidth + 'px';
        }
    }
    e.preventDefault();
});

document.addEventListener('mouseup', function() {
    dragState.active = false;
    dragState.direction = '';
    dragState.targetBox = null;
});

// mm:ss 형식의 문자열을 초 단위로 변환하는 함수 (예: "02:30" -> 150초)
function parseTimeString(timeStr) {
    const parts = timeStr.split(':');
    if (parts.length === 2) {
        const minutes = parseInt(parts[0], 10);
        const seconds = parseFloat(parts[1]);
        if (isNaN(minutes) || isNaN(seconds)) {
            return 0;
        }
        return minutes * 60 + seconds;
    }
    return 0;
}

// OCR 시작
// POST start_ocr를 호출하면 task_id를 받고, WebSocket 업데이트로 진행률이 표시됨.
startOcrBtn.addEventListener('click', async function() {
    const regionsPayload = getEnabledRegions();
    
    // 새롭게 추가된 시간대 값 읽기 (초 단위)
    const startTimeInput = document.getElementById('startTimeInput');
    const endTimeInput = document.getElementById('endTimeInput');
    let startTime = parseTimeString(startTimeInput.value);
    let endTime = parseTimeString(endTimeInput.value);
    
    if (!vfilename) {
        alert('비디오 파일을 선택해주세요.');
        return;
    }
    if (!regionsPayload.length) {
        alert('OCR 영역을 최소 1개 이상 지정해주세요.');
        return;
    }
    let formData = new FormData();
    formData.append('video_filename', vfilename);
    formData.append('regions', JSON.stringify(regionsPayload));
    // 첫 번째 영역은 기존 백엔드와의 호환을 위해 그대로 전달
    formData.append('x', regionsPayload[0].x);
    formData.append('y', regionsPayload[0].y);
    formData.append('width', regionsPayload[0].width);
    formData.append('height', regionsPayload[0].height);
    if (startTime != 0) {
        formData.append('start_time', startTime);
    }
    if (endTime != 0) {
        formData.append('end_time', endTime);
    }
    
    try {
        let response = await fetch('/start_ocr/', {
            method: 'POST',
            body: formData
        });
        
        // 응답이 정상(200~299) 범위가 아닌 경우
        if (!response.ok) {
            let errorData = await response.json();
            // 응답 메시지가 {"detail": "오류 메시지"} 형식이므로 detail 필드 사용
            alert('OCR 작업 시작 중 오류 발생: ' + errorData.detail);
            return;
        }
        
        // 정상 응답일 경우
        let data = await response.json();
        console.log(data.task_id)
        switchToTaskListView();
    } catch (err) {
        // 네트워크 오류 등 예외 발생 시
        alert('OCR 작업 시작 중 오류 발생: ' + err.message);
    }
});
