(() => {
    const taskId = document.body.dataset.taskId;
    const videoEl = document.getElementById('preview-video');
    const overlayEl = document.getElementById('subtitle-overlay');
    const subtitleListEl = document.getElementById('subtitle-list');
    const sliderPanelEl = document.getElementById('slider-panel');
    const statusEl = document.getElementById('editor-status');
    const fileNameEl = document.getElementById('editor-file-name');
    const segmentCountEl = document.getElementById('segment-count');
    const metricsSummaryEl = document.getElementById('metrics-summary');
    const srtFileNameEl = document.getElementById('srt-file-name');
    const resetBtn = document.getElementById('reset-params-btn');
    const mergeBtn = document.getElementById('merge-save-btn');

    const sliderConfigs = [
        {
            key: 'duplicate_gap_sec',
            label: '같은 자막 재연결 간격',
            min: 0.0,
            max: 5.0,
            step: 0.1,
            caption: '같은 문장이 잠깐 끊겼을 때 다시 하나로 붙일 수 있는 최대 시간입니다.',
            format: value => `${Number(value).toFixed(1)}초`,
        },
        {
            key: 'contained_gap_sec',
            label: '포함 파편 흡수 간격',
            min: 0.0,
            max: 3.0,
            step: 0.1,
            caption: '긴 문장 안에 포함되는 짧은 OCR 파편을 흡수할 최대 시간입니다.',
            format: value => `${Number(value).toFixed(1)}초`,
        },
        {
            key: 'min_contained_key_len',
            label: '포함 파편 최소 글자수',
            min: 2,
            max: 12,
            step: 1,
            caption: '너무 짧아 의미가 불안정한 포함 파편은 병합 대상에서 제외합니다.',
            format: value => `${Number(value)}자`,
            integer: true,
        },
        {
            key: 'similar_threshold',
            label: '고유사 병합 임계값',
            min: 0.88,
            max: 1.0,
            step: 0.005,
            caption: 'OCR 한두 글자 차이를 같은 자막으로 볼 유사도 기준입니다.',
            format: value => Number(value).toFixed(3),
        },
        {
            key: 'min_similar_key_len',
            label: '고유사 병합 최소 글자수',
            min: 4,
            max: 24,
            step: 1,
            caption: '짧은 문장끼리는 우연히 비슷할 수 있어 이 길이 이상만 유사 병합합니다.',
            format: value => `${Number(value)}자`,
            integer: true,
        },
        {
            key: 'similar_length_ratio',
            label: '고유사 길이 비율',
            min: 0.5,
            max: 1.0,
            step: 0.05,
            caption: '두 후보의 길이가 지나치게 다르면 유사도가 높아도 병합하지 않습니다.',
            format: value => Number(value).toFixed(2),
        },
        {
            key: 'min_duration_sec',
            label: '최소 자막 길이',
            min: 0.0,
            max: 2.0,
            step: 0.1,
            caption: '후처리 후 이보다 짧은 잔여 세그먼트는 노이즈로 보고 제거합니다.',
            format: value => `${Number(value).toFixed(1)}초`,
        },
        {
            key: 'postprocess_passes',
            label: '반복 정리 횟수',
            min: 1,
            max: 5,
            step: 1,
            caption: '동일/포함/고유사 병합을 반복 적용하는 최대 횟수입니다.',
            format: value => `${Number(value)}회`,
            integer: true,
        },
    ];

    let defaultParams = {};
    let currentParams = {};
    let previewSegments = [];
    let rowByIndex = new Map();
    let previewTimer = null;
    let previewAbortController = null;
    let activeIndexKey = '';
    let lastScrolledIndex = null;

    function setStatus(message, tone = 'neutral') {
        // 상태 문구와 색상만 갱신해 레이아웃 흔들림을 줄입니다.
        statusEl.textContent = message;
        statusEl.dataset.tone = tone;
    }

    function numericValue(config, value) {
        // 슬라이더 값은 API 계약에 맞춰 정수와 실수를 분리해 보냅니다.
        return config.integer ? Number.parseInt(value, 10) : Number(value);
    }

    function formatTimestamp(seconds) {
        // 리스트에는 SRT와 같은 시간 표기를 보여줍니다.
        const safeSeconds = Math.max(0, Number(seconds) || 0);
        const totalMs = Math.round(safeSeconds * 1000);
        const ms = totalMs % 1000;
        const totalSec = Math.floor(totalMs / 1000);
        const sec = totalSec % 60;
        const totalMin = Math.floor(totalSec / 60);
        const min = totalMin % 60;
        const hour = Math.floor(totalMin / 60);
        return `${String(hour).padStart(2, '0')}:${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
    }

    async function fetchJson(url, options = {}) {
        // JSON API 오류 응답의 detail을 상태 문구에 활용할 수 있게 보존합니다.
        const response = await fetch(url, options);
        let payload = null;
        try {
            payload = await response.json();
        } catch (_error) {
            payload = null;
        }
        if (!response.ok) {
            const detail = payload && payload.detail ? payload.detail : response.statusText;
            throw new Error(Array.isArray(detail) ? JSON.stringify(detail) : String(detail));
        }
        return payload;
    }

    function renderSliders() {
        // 메타데이터의 기본값을 기준으로 모든 슬라이더를 생성합니다.
        sliderPanelEl.textContent = '';
        for (const config of sliderConfigs) {
            const row = document.createElement('div');
            row.className = 'slider-row';

            const main = document.createElement('div');
            main.className = 'slider-main';

            const labelWrap = document.createElement('div');
            labelWrap.className = 'slider-label-row';

            const label = document.createElement('label');
            label.className = 'slider-label';
            label.htmlFor = `slider-${config.key}`;
            label.textContent = config.label;

            const valueEl = document.createElement('span');
            valueEl.className = 'slider-value';
            valueEl.textContent = config.format(currentParams[config.key]);

            labelWrap.appendChild(label);
            labelWrap.appendChild(valueEl);

            const input = document.createElement('input');
            input.type = 'range';
            input.id = `slider-${config.key}`;
            input.className = 'custom-range merge-slider';
            input.min = String(config.min);
            input.max = String(config.max);
            input.step = String(config.step);
            input.value = String(currentParams[config.key]);
            input.addEventListener('input', () => {
                currentParams[config.key] = numericValue(config, input.value);
                valueEl.textContent = config.format(currentParams[config.key]);
                schedulePreviewRefresh();
            });

            const caption = document.createElement('div');
            caption.className = 'slider-caption';
            caption.textContent = config.caption;
            caption.title = config.caption;

            main.appendChild(labelWrap);
            main.appendChild(input);
            row.appendChild(main);
            row.appendChild(caption);
            sliderPanelEl.appendChild(row);
        }
    }

    function renderSubtitles(segments) {
        // 프리뷰 결과를 왼쪽 자막 리스트에 반영합니다.
        subtitleListEl.textContent = '';
        rowByIndex = new Map();

        const fragment = document.createDocumentFragment();
        for (const segment of segments) {
            const row = document.createElement('div');
            row.className = 'subtitle-row';
            row.dataset.index = String(segment.index);
            row.tabIndex = 0;

            const time = document.createElement('div');
            time.className = 'subtitle-time';
            time.textContent = `${formatTimestamp(segment.start)} --> ${formatTimestamp(segment.end)}`;

            const text = document.createElement('div');
            text.className = 'subtitle-text';
            text.textContent = segment.text;

            row.appendChild(time);
            row.appendChild(text);
            row.addEventListener('dblclick', () => {
                videoEl.currentTime = Math.max(0, Number(segment.start) + 0.05);
                videoEl.focus();
                updateActiveSubtitle();
            });
            row.addEventListener('keydown', event => {
                if (event.key === 'Enter') {
                    videoEl.currentTime = Math.max(0, Number(segment.start) + 0.05);
                    videoEl.focus();
                    updateActiveSubtitle();
                }
            });

            rowByIndex.set(segment.index, row);
            fragment.appendChild(row);
        }

        subtitleListEl.appendChild(fragment);
        segmentCountEl.textContent = `${segments.length}개`;
    }

    function renderMetrics(metrics) {
        // 병합 결과 규모를 한 줄로 보여줍니다.
        if (!metrics) {
            metricsSummaryEl.textContent = '세그먼트 0개';
            return;
        }
        metricsSummaryEl.textContent = `세그먼트 ${metrics.segment_count}개 · 평균 ${metrics.average_duration_sec}초 · 총 ${metrics.total_duration_sec}초`;
    }

    function updateActiveSubtitle() {
        // 영상 시간에 겹치는 cue를 overlay와 리스트 highlight에 동시에 반영합니다.
        const currentTime = Number(videoEl.currentTime) || 0;
        const activeSegments = previewSegments.filter(segment => (
            currentTime + 0.001 >= Number(segment.start) &&
            currentTime <= Number(segment.end) + 0.001
        ));
        const activeIndexes = activeSegments.map(segment => segment.index);
        const nextIndexKey = activeIndexes.join(',');
        if (nextIndexKey === activeIndexKey) {
            return;
        }
        activeIndexKey = nextIndexKey;

        for (const [index, row] of rowByIndex.entries()) {
            row.classList.toggle('active', activeIndexes.includes(index));
        }

        overlayEl.textContent = activeSegments.map(segment => segment.text).join('\n');

        if (activeIndexes.length > 0 && activeIndexes[0] !== lastScrolledIndex) {
            const activeRow = rowByIndex.get(activeIndexes[0]);
            if (activeRow) {
                activeRow.scrollIntoView({ block: 'nearest' });
                lastScrolledIndex = activeIndexes[0];
            }
        }
    }

    function applyPreviewPayload(payload) {
        // 서버에서 받은 segment와 metrics를 화면 상태에 반영합니다.
        previewSegments = payload.segments || [];
        currentParams = { ...currentParams, ...(payload.params || {}) };
        activeIndexKey = '';
        renderSubtitles(previewSegments);
        renderMetrics(payload.metrics);
        updateActiveSubtitle();
    }

    async function refreshPreview() {
        // 이전 미리보기 요청을 취소하고 최신 슬라이더 값만 계산합니다.
        if (previewAbortController) {
            previewAbortController.abort();
        }
        previewAbortController = new AbortController();
        const controller = previewAbortController;
        setStatus('미리보기 갱신 중', 'loading');

        try {
            const payload = await fetchJson(`/api/subtitle-editor/${encodeURIComponent(taskId)}/preview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ params: currentParams }),
                signal: controller.signal,
            });
            if (controller !== previewAbortController) {
                return;
            }
            applyPreviewPayload(payload);
            setStatus('미리보기 갱신 완료', 'success');
        } catch (error) {
            if (error.name === 'AbortError') {
                return;
            }
            setStatus(`미리보기 실패: ${error.message}`, 'error');
        }
    }

    function schedulePreviewRefresh() {
        // 슬라이더 입력 중에는 400ms 디바운스로 서버 계산을 줄입니다.
        window.clearTimeout(previewTimer);
        setStatus('미리보기 갱신 대기', 'loading');
        previewTimer = window.setTimeout(() => {
            refreshPreview();
        }, 400);
    }

    async function saveMergedSubtitle() {
        // 병합 버튼에서만 실제 SRT 파일을 덮어씁니다.
        if (previewAbortController) {
            previewAbortController.abort();
            previewAbortController = null;
        }
        mergeBtn.disabled = true;
        setStatus('병합 저장 중', 'loading');
        try {
            const payload = await fetchJson(`/api/subtitle-editor/${encodeURIComponent(taskId)}/merge`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ params: currentParams }),
            });
            applyPreviewPayload(payload);
            if (payload.srt_filename) {
                srtFileNameEl.textContent = payload.srt_filename;
            }
            setStatus('병합 저장 완료', 'success');
        } catch (error) {
            setStatus(`병합 저장 실패: ${error.message}`, 'error');
        } finally {
            mergeBtn.disabled = false;
        }
    }

    async function loadMetadata() {
        // 작업 메타데이터를 불러온 뒤 첫 미리보기를 계산합니다.
        setStatus('메타데이터 로딩 중', 'loading');
        const metadata = await fetchJson(`/api/subtitle-editor/${encodeURIComponent(taskId)}`);
        defaultParams = { ...metadata.default_params };
        currentParams = { ...metadata.current_params };
        fileNameEl.textContent = metadata.video_filename || '';
        videoEl.src = metadata.video_url;
        srtFileNameEl.textContent = metadata.srt_filename || 'SRT 없음';
        mergeBtn.disabled = !metadata.can_merge;
        renderSliders();
        await refreshPreview();
    }

    resetBtn.addEventListener('click', () => {
        // 기본값 복원은 파일을 쓰지 않고 프리뷰만 다시 계산합니다.
        currentParams = { ...defaultParams };
        renderSliders();
        schedulePreviewRefresh();
    });

    mergeBtn.addEventListener('click', () => {
        saveMergedSubtitle();
    });

    videoEl.addEventListener('timeupdate', updateActiveSubtitle);
    videoEl.addEventListener('seeking', updateActiveSubtitle);
    videoEl.addEventListener('seeked', updateActiveSubtitle);

    loadMetadata().catch(error => {
        setStatus(`초기화 실패: ${error.message}`, 'error');
        mergeBtn.disabled = true;
        resetBtn.disabled = true;
    });
})();
