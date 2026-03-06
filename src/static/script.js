const video = document.getElementById('video');
const startOcrBtn = document.getElementById('start-ocr-btn');
const boundingBox = document.getElementById('bounding-box');
const handles = document.querySelectorAll('.handle');
const maskBox = document.getElementById('mask-box');

const fileList = document.getElementById('file-list');
const currentPathElem = document.getElementById('current-path');
const taskListView = document.getElementById('task-list-view');
const ocrCreationView = document.getElementById('ocr-creation-view');
const taskListTableBody = document.querySelector('#task-list tbody');
const videoContainer = document.getElementById('video-container');
const stopAllBtn = document.getElementById('stop-all-btn');

const newOcrBtn = document.getElementById('new-ocr-btn');
const backToListBtn = document.getElementById('back-to-list-btn');

const fullScreenOcrToggle = document.getElementById('fullScreenOcrToggle');
const maskToggle = document.getElementById('maskToggle');
const clearMaskBtn = document.getElementById('clearMaskBtn');
const maskControls = document.getElementById('mask-controls');

let vfilename = null;
let currentPath = '';

const taskStatusMap = {};
let isTaskRunning = false;

let dragging = false;
let dragDirection = '';
let startY = 0;
let startX = 0;
let startHeight = 0;
let startWidth = 0;
let startTop = 0;
let startLeft = 0;

let maskDrawing = false;
let maskStartX = 0;
let maskStartY = 0;

const MIN_BOX_SIZE = 20;
const MIN_MASK_SIZE = 10;
const VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.mpg', '.mpeg', '.wmv'];

// WebSocket 연결 (단일 연결)
const ws = new WebSocket(`ws://${location.host}/ws/tasks`);

ws.onopen = () => {
    console.log("WebSocket 연결 성공");
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    // message는 { task_id, progress, status, estimated_completion, error, ... } 형태
    updateTaskRow(message);
};

ws.onerror = (err) => {
    console.error("WebSocket 오류:", err);
};

// 작업 목록 초기 로드
function loadTaskList() {
    return fetch('/tasks/')
        .then((response) => response.json())
        .then((data) => {
            // data는 tasks 딕셔너리 형태 { task_id: { ... }, ... }
            for (const taskId in data) {
                const task = data[taskId];
                task.task_id = taskId;
                updateTaskRow(task);
            }
        })
        .catch((err) => console.error(err));
}

loadTaskList();

// --- 함수 정의 ---
function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function encodePath(path) {
    return path.split('/').map(encodeURIComponent).join('/');
}

function isVideoFile(path) {
    const lower = path.toLowerCase();
    return VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function getVideoDisplayWidth() {
    return video.clientWidth || 0;
}

function getVideoDisplayHeight() {
    return video.clientHeight || 0;
}

function getPointInVideo(clientX, clientY) {
    const rect = video.getBoundingClientRect();
    if (!rect.width || !rect.height) {
        return null;
    }

    const x = clamp(clientX - rect.left, 0, rect.width);
    const y = clamp(clientY - rect.top, 0, rect.height);
    return { x, y };
}

function setBoundingBoxRect(x, y, width, height) {
    boundingBox.style.left = `${x}px`;
    boundingBox.style.top = `${y}px`;
    boundingBox.style.width = `${Math.max(width, MIN_BOX_SIZE)}px`;
    boundingBox.style.height = `${Math.max(height, MIN_BOX_SIZE)}px`;
}

function resetBoundingBoxToFullVideo() {
    const width = getVideoDisplayWidth();
    const height = getVideoDisplayHeight();
    setBoundingBoxRect(0, 0, width, height);
}

function clearMaskBox() {
    maskBox.style.display = 'none';
    maskBox.style.left = '0px';
    maskBox.style.top = '0px';
    maskBox.style.width = '0px';
    maskBox.style.height = '0px';
}

function setMaskRect(x, y, width, height) {
    maskBox.style.display = 'block';
    maskBox.style.left = `${x}px`;
    maskBox.style.top = `${y}px`;
    maskBox.style.width = `${width}px`;
    maskBox.style.height = `${height}px`;
}

function hasMaskRect() {
    return maskBox.style.display === 'block' && maskBox.offsetWidth >= MIN_MASK_SIZE && maskBox.offsetHeight >= MIN_MASK_SIZE;
}

function isFullScreenOcrEnabled() {
    return Boolean(fullScreenOcrToggle.checked);
}

function isMaskDrawingEnabled() {
    return isFullScreenOcrEnabled() && Boolean(maskToggle.checked);
}

function updateBoundingBoxInteraction() {
    const locked = isFullScreenOcrEnabled();
    boundingBox.classList.toggle('locked', locked);
    handles.forEach((handle) => {
        handle.style.display = locked ? 'none' : 'block';
    });
    if (locked) {
        resetBoundingBoxToFullVideo();
    }
}

function updateMaskControls() {
    const fullScreenEnabled = isFullScreenOcrEnabled();

    maskToggle.disabled = !fullScreenEnabled;
    if (!fullScreenEnabled) {
        maskToggle.checked = false;
        clearMaskBox();
    }

    const drawingEnabled = isMaskDrawingEnabled();
    clearMaskBtn.disabled = !drawingEnabled;

    videoContainer.classList.toggle('mask-draw-enabled', drawingEnabled);
    maskControls.classList.toggle('mask-disabled', !fullScreenEnabled);

    if (!drawingEnabled) {
        maskDrawing = false;
    }
}

function loadDirectory(path = '') {
    fetch(`/browse/?path=${encodeURIComponent(path)}`)
        .then((resp) => resp.json())
        .then((data) => {
            currentPath = data.path;
            currentPathElem.textContent = '/' + (currentPath || '');
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

            data.entries.forEach((entry) => {
                const item = document.createElement('li');
                item.className = 'list-group-item list-group-item-action browser-item';

                const icon = document.createElement('span');
                icon.className = 'item-icon';
                if (entry.is_dir) {
                    icon.textContent = '\uD83D\uDCC1';
                } else {
                    icon.textContent = isVideoFile(entry.name) ? '\uD83C\uDFAC' : '\uD83D\uDCC4';
                }

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
        })
        .catch((err) => {
            console.error('Failed to browse directory:', err);
        });
}

function selectVideo(path) {
    const isVideo = isVideoFile(path);
    if (!isVideo) {
        alert('비디오 파일을 선택해주세요.');
        return;
    }

    vfilename = path;
    videoContainer.style.display = 'block';
    video.src = `/videos/${encodePath(path)}`;
    video.load();

    video.onloadedmetadata = function () {
        videoContainer.style.width = `${video.clientWidth}px`;
        videoContainer.style.height = `${video.clientHeight}px`;

        resetBoundingBoxToFullVideo();
        clearMaskBox();
        updateBoundingBoxInteraction();
        updateMaskControls();
    };
}

function updateRunningStatus(taskId, status) {
    if (typeof status === 'undefined') {
        delete taskStatusMap[taskId];
    } else {
        taskStatusMap[taskId] = status;
    }

    isTaskRunning = Object.values(taskStatusMap).some((s) => s === 'running');
}

function setActionButtons(row, status, taskId) {
    const cell = row.querySelector('.action-cell');
    cell.innerHTML = '';

    if (status === 'running' || status === 'waiting') {
        const stopBtn = document.createElement('button');
        stopBtn.className = 'btn btn-warning btn-sm stop-btn';
        stopBtn.innerText = '중지';
        stopBtn.onclick = () => stopTask(taskId);
        cell.appendChild(stopBtn);
        return;
    }

    if (status === 'stopped' || status === 'stopping' || status === 'error') {
        const resumeBtn = document.createElement('button');
        resumeBtn.className = 'btn btn-primary btn-sm resume-btn mr-1';
        resumeBtn.innerText = '재개';
        resumeBtn.onclick = () => resumeTask(taskId);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn btn-danger btn-sm delete-btn';
        deleteBtn.innerText = '삭제';
        deleteBtn.onclick = () => deleteTask(taskId);

        cell.appendChild(resumeBtn);
        cell.appendChild(deleteBtn);
        return;
    }

    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'btn btn-danger btn-sm delete-btn';
    deleteBtn.innerText = '삭제';
    deleteBtn.onclick = () => deleteTask(taskId);
    cell.appendChild(deleteBtn);
}

function updateTaskRow(task) {
    const taskId = task.task_id;
    const progress = task.progress || 0;
    const status = task.status || '';

    let statusText = status;
    if (status === 'waiting') {
        statusText = 'waiting';
    }
    if (task.error) {
        statusText += `: ${task.error}`;
    }

    const estimated = typeof task.estimated_completion !== 'undefined' ? task.estimated_completion : 'TBD';
    const videoFile = task.video_filename || '';

    let row = document.getElementById(`task-row-${taskId}`);
    if (!row) {
        row = document.createElement('tr');
        row.id = `task-row-${taskId}`;
        row.innerHTML = `
            <td class="video-file"></td>
            <td class="progress-cell">
                <div class="progress">
                    <div class="progress-bar" role="progressbar" style="width: ${progress}%;" aria-valuenow="${progress}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
            </td>
            <td class="status-cell"></td>
            <td class="estimated-cell"></td>
            <td class="action-cell"></td>
        `;
        taskListTableBody.appendChild(row);
    }

    const videoCell = row.querySelector('.video-file');
    videoCell.textContent = videoFile;
    videoCell.title = videoFile;

    const progressBar = row.querySelector('.progress-bar');
    progressBar.style.width = `${progress}%`;
    progressBar.setAttribute('aria-valuenow', String(progress));

    row.querySelector('.status-cell').innerText = statusText;
    row.querySelector('.estimated-cell').innerText = estimated;

    setActionButtons(row, status, taskId);
    updateRunningStatus(taskId, status);
}

function stopTask(taskId) {
    const formData = new FormData();
    formData.append('task_id', taskId);

    fetch('/stop_ocr/', {
        method: 'POST',
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => console.log(data.detail))
        .catch((err) => console.error(err));
}

function resumeTask(taskId) {
    const formData = new FormData();
    formData.append('task_id', taskId);

    fetch('/resume_ocr/', {
        method: 'POST',
        body: formData,
    })
        .then(async (response) => {
            const data = await response.json();
            if (!response.ok) {
                if (data.detail === 'RESUME_NOT_READY') {
                    alert('Task is not ready to resume yet.');
                } else {
                    alert(`OCR resume failed: ${data.detail}`);
                }
                return;
            }

            console.log(data.detail);
        })
        .catch((err) => console.error(err));
}

function stopAllTasks() {
    fetch('/stop_all_ocr/', {
        method: 'POST',
    })
        .then(async (response) => {
            const data = await response.json();
            if (!response.ok) {
                alert(`전체 중지 실패: ${data.detail}`);
                return;
            }

            console.log(data.detail);
        })
        .catch((err) => console.error(err));
}

function deleteTask(taskId) {
    fetch(`/tasks/${taskId}`, {
        method: 'DELETE',
    })
        .then((response) => response.json())
        .then(() => {
            const row = document.getElementById(`task-row-${taskId}`);
            if (row) {
                row.remove();
            }
            delete taskStatusMap[taskId];
            updateRunningStatus(taskId, undefined);
        })
        .catch((err) => console.error(err));
}

function switchToOcrCreationView() {
    taskListView.style.display = 'none';
    ocrCreationView.style.display = 'block';
    loadDirectory('');

    videoContainer.style.display = 'none';
    vfilename = null;

    fullScreenOcrToggle.checked = false;
    maskToggle.checked = false;
    clearMaskBox();
    updateBoundingBoxInteraction();
    updateMaskControls();
}

function switchToTaskListView() {
    ocrCreationView.style.display = 'none';
    taskListView.style.display = 'block';
}

// 이벤트: "새 OCR 작업 추가" 버튼 -> OCR 생성 뷰로 전환
newOcrBtn.addEventListener('click', () => {
    switchToOcrCreationView();
});

// 이벤트: "작업 큐로 돌아가기" 버튼 -> 작업 목록 뷰로 전환
backToListBtn.addEventListener('click', () => {
    switchToTaskListView();
});

stopAllBtn.addEventListener('click', () => {
    stopAllTasks();
});

// 이벤트: 전체 화면 OCR 토글 변경
fullScreenOcrToggle.addEventListener('change', () => {
    updateBoundingBoxInteraction();
    updateMaskControls();
});

maskToggle.addEventListener('change', () => {
    if (!isMaskDrawingEnabled()) {
        clearMaskBox();
    }
    updateMaskControls();
});

clearMaskBtn.addEventListener('click', () => {
    clearMaskBox();
});




// 드래그 핸들 이벤트 처리
handles.forEach((handle) => {
    handle.addEventListener('mousedown', (e) => {
        if (isFullScreenOcrEnabled()) {
            return;
        }

        dragging = true;
        dragDirection =
            e.target.classList.contains('top') ? 'top' :
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

videoContainer.addEventListener('mousedown', (e) => {
    if (!isMaskDrawingEnabled()) {
        return;
    }

    if (e.button !== 0) {
        return;
    }

    if (e.target.closest('.handle')) {
        return;
    }

    const point = getPointInVideo(e.clientX, e.clientY);
    if (!point) {
        return;
    }

    maskDrawing = true;
    maskStartX = point.x;
    maskStartY = point.y;
    setMaskRect(maskStartX, maskStartY, 1, 1);
    e.preventDefault();
});

function updateBoundingBoxOnDrag(clientX, clientY) {
    const dy = clientY - startY;
    const dx = clientX - startX;

    if (dragDirection === 'top') {
        const maxShift = startHeight - MIN_BOX_SIZE;
        const appliedShift = clamp(dy, -startTop, maxShift);
        boundingBox.style.top = `${startTop + appliedShift}px`;
        boundingBox.style.height = `${startHeight - appliedShift}px`;
        return;
    }

    if (dragDirection === 'bottom') {
        const maxHeight = getVideoDisplayHeight() - startTop;
        const newHeight = clamp(startHeight + dy, MIN_BOX_SIZE, maxHeight);
        boundingBox.style.height = `${newHeight}px`;
        return;
    }

    if (dragDirection === 'left') {
        const maxShift = startWidth - MIN_BOX_SIZE;
        const appliedShift = clamp(dx, -startLeft, maxShift);
        boundingBox.style.left = `${startLeft + appliedShift}px`;
        boundingBox.style.width = `${startWidth - appliedShift}px`;
        return;
    }

    if (dragDirection === 'right') {
        const maxWidth = getVideoDisplayWidth() - startLeft;
        const newWidth = clamp(startWidth + dx, MIN_BOX_SIZE, maxWidth);
        boundingBox.style.width = `${newWidth}px`;
    }
}

function updateMaskOnDraw(clientX, clientY) {
    const point = getPointInVideo(clientX, clientY);
    if (!point) {
        return;
    }

    const left = Math.min(maskStartX, point.x);
    const top = Math.min(maskStartY, point.y);
    const width = Math.abs(point.x - maskStartX);
    const height = Math.abs(point.y - maskStartY);

    setMaskRect(left, top, width, height);
}

document.addEventListener('mousemove', (e) => {
    if (dragging) {
        updateBoundingBoxOnDrag(e.clientX, e.clientY);
        e.preventDefault();
    }

    if (maskDrawing) {
        updateMaskOnDraw(e.clientX, e.clientY);
        e.preventDefault();
    }
});

document.addEventListener('mouseup', () => {
    dragging = false;
    dragDirection = '';

    if (maskDrawing) {
        maskDrawing = false;
        if (!hasMaskRect()) {
            clearMaskBox();
        }
    }
});

// mm:ss 형식의 문자열을 초 단위로 변환하는 함수 (예: "02:30" -> 150초)
function parseTimeString(timeStr) {
    const parts = String(timeStr || '').split(':');
    if (parts.length !== 2) {
        return 0;
    }

    const minutes = parseInt(parts[0], 10);
    const seconds = parseFloat(parts[1]);
    if (Number.isNaN(minutes) || Number.isNaN(seconds)) {
        return 0;
    }

    return minutes * 60 + seconds;
}

function getBoundingBoxInSourcePixels(scaleX, scaleY) {
    const videoRect = video.getBoundingClientRect();
    const boxRect = boundingBox.getBoundingClientRect();

    const x = Math.round((boxRect.left - videoRect.left) * scaleX);
    const y = Math.round((boxRect.top - videoRect.top) * scaleY);
    const width = Math.round(boxRect.width * scaleX);
    const height = Math.round(boxRect.height * scaleY);

    return { x, y, width, height };
}

function getMaskBoxInSourcePixels(scaleX, scaleY) {
    if (!hasMaskRect()) {
        return null;
    }

    const videoRect = video.getBoundingClientRect();
    const boxRect = maskBox.getBoundingClientRect();

    const x = Math.round((boxRect.left - videoRect.left) * scaleX);
    const y = Math.round((boxRect.top - videoRect.top) * scaleY);
    const width = Math.round(boxRect.width * scaleX);
    const height = Math.round(boxRect.height * scaleY);

    return {
        x: Math.max(0, x),
        y: Math.max(0, y),
        width: Math.max(0, width),
        height: Math.max(0, height),
    };
}

// OCR 시작
// POST start_ocr를 호출하면 task_id를 받고, WebSocket 업데이트로 진행률이 표시됨.
startOcrBtn.addEventListener('click', async () => {
    if (!vfilename) {
        alert('Please select a video file.');
        return;
    }

    if (!video.videoWidth || !video.videoHeight || !video.clientWidth || !video.clientHeight) {
        alert('Video metadata is not ready yet. Please try again.');
        return;
    }

    // 새롭게 추가된 시간대 값 읽기 (초 단위)
    const startTimeInput = document.getElementById('startTimeInput');
    const endTimeInput = document.getElementById('endTimeInput');
    const startTime = parseTimeString(startTimeInput.value);
    const endTime = parseTimeString(endTimeInput.value);

    const scaleX = video.videoWidth / video.clientWidth;
    const scaleY = video.videoHeight / video.clientHeight;

    const fullScreenOcr = isFullScreenOcrEnabled();

    let x;
    let y;
    let width;
    let height;

    if (fullScreenOcr) {
        x = 0;
        y = 0;
        width = video.videoWidth;
        height = video.videoHeight;
    } else {
        const rect = getBoundingBoxInSourcePixels(scaleX, scaleY);
        x = rect.x;
        y = rect.y;
        width = rect.width;
        height = rect.height;
    }

    const formData = new FormData();
    formData.append('video_filename', vfilename);
    formData.append('x', String(x));
    formData.append('y', String(y));
    formData.append('width', String(width));
    formData.append('height', String(height));
    formData.append('full_screen_ocr', fullScreenOcr ? 'true' : 'false');

    if (startTime !== 0) {
        formData.append('start_time', String(startTime));
    }
    if (endTime !== 0) {
        formData.append('end_time', String(endTime));
    }

    if (fullScreenOcr && maskToggle.checked) {
        const maskRect = getMaskBoxInSourcePixels(scaleX, scaleY);
        if (!maskRect || maskRect.width <= 0 || maskRect.height <= 0) {
            alert('Mask is enabled, but no mask area was selected. Drag on video to set the area.');
            return;
        }

        formData.append('mask_x', String(maskRect.x));
        formData.append('mask_y', String(maskRect.y));
        formData.append('mask_width', String(maskRect.width));
        formData.append('mask_height', String(maskRect.height));
    }

    try {
        const response = await fetch('/start_ocr/', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            // 응답 메시지가 {"detail": "오류 메시지"} 형식이므로 detail 필드 사용
            alert(`OCR 작업 시작 중 오류 발생: ${errorData.detail}`);
            return;
        }

        // 정상 응답일 경우
        const data = await response.json();
        console.log(data.task_id);
        await loadTaskList();
        switchToTaskListView();
    } catch (err) {
        // 네트워크 오류 등 예외 발생 시
        alert(`OCR 작업 시작 중 오류 발생: ${err.message}`);
    }
});

updateBoundingBoxInteraction();
updateMaskControls();
