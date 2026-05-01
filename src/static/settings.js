document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('settings-form');
    const dockerToggle = document.getElementById('docker-enabled');
    const dockerUrlInput = document.getElementById('docker-url');
    const detectorDockerNameSelect = document.getElementById('detector-docker-name');
    const recognizerDockerNameSelect = document.getElementById('recognizer-docker-name');
    const refreshDockerListButton = document.getElementById('refresh-docker-list');
    const detectorBaseUrlInput = document.getElementById('detector-llm-base-url');
    const detectorModelInput = document.getElementById('detector-llm-model');
    const recognizerBaseUrlInput = document.getElementById('recognizer-llm-base-url');
    const recognizerModelInput = document.getElementById('recognizer-llm-model');
    const llmApiKeyInput = document.getElementById('llm-api-key');
    const kafkaToggle = document.getElementById('kafka-enabled');
    const kafkaUrlInput = document.getElementById('kafka-url');
    const feedback = document.getElementById('settings-feedback');
    const saveButton = document.getElementById('save-button');
    const dockerControls = [
        dockerUrlInput,
        detectorDockerNameSelect,
        recognizerDockerNameSelect,
        refreshDockerListButton,
    ];
    const dockerSelects = [detectorDockerNameSelect, recognizerDockerNameSelect];

    const showFeedback = (variant, message) => {
        feedback.className = `alert alert-${variant}`;
        feedback.textContent = message;
        feedback.classList.remove('d-none');
    };

    const clearFeedback = () => {
        feedback.classList.add('d-none');
        feedback.textContent = '';
    };

    const toggleKafkaUrl = (enabled) => {
        kafkaUrlInput.disabled = !enabled;
        kafkaUrlInput.parentElement.classList.toggle('disabled-field', !enabled);
    };

    const toggleDockerFields = (enabled) => {
        dockerControls.forEach((control) => {
            control.disabled = !enabled;
            const group = control.closest('.form-group');
            if (group) {
                group.classList.toggle('disabled-field', !enabled);
            }
        });
    };

    const setDockerListLoading = (isLoading) => {
        const enabled = dockerToggle.checked;
        dockerSelects.forEach((select) => {
            select.disabled = isLoading || !enabled;
        });
        refreshDockerListButton.disabled = isLoading || !enabled;
    };

    const renderDockerSelectOptions = (select, containers, selectedName) => {
        select.innerHTML = '';

        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = '컨테이너를 선택하세요';
        select.appendChild(placeholder);

        const knownNames = new Set();
        containers.forEach((containerName) => {
            const option = document.createElement('option');
            option.value = containerName;
            option.textContent = containerName;
            select.appendChild(option);
            knownNames.add(containerName);
        });

        if (selectedName && !knownNames.has(selectedName)) {
            const missingOption = document.createElement('option');
            missingOption.value = selectedName;
            missingOption.textContent = `${selectedName} (현재 설정 - 목록에 없음)`;
            select.appendChild(missingOption);
        }

        if (selectedName) {
            select.value = selectedName;
        }
    };

    const renderDockerOptions = (containers, selectedNames) => {
        renderDockerSelectOptions(detectorDockerNameSelect, containers, selectedNames.detector || '');
        renderDockerSelectOptions(recognizerDockerNameSelect, containers, selectedNames.recognizer || '');
    };

    const getSelectedDockerNames = () => ({
        detector: detectorDockerNameSelect.value.trim(),
        recognizer: recognizerDockerNameSelect.value.trim(),
    });

    const fetchDockerContainers = async (selectedNames = getSelectedDockerNames()) => {
        if (!dockerToggle.checked) {
            renderDockerOptions([], selectedNames);
            return;
        }

        const dockerUrl = dockerUrlInput.value.trim();
        if (!dockerUrl) {
            renderDockerOptions([], selectedNames);
            return;
        }

        setDockerListLoading(true);
        try {
            const res = await fetch(`/api/docker/containers?docker_url=${encodeURIComponent(dockerUrl)}`);
            const responseBody = await res.json().catch(() => ({}));
            if (!res.ok) {
                throw new Error(responseBody.detail || '컨테이너 목록을 불러오지 못했습니다.');
            }

            const containers = Array.isArray(responseBody.containers) ? responseBody.containers : [];
            renderDockerOptions(containers, selectedNames);
        } catch (error) {
            renderDockerOptions([], selectedNames);
            showFeedback('warning', error.message || '컨테이너 목록을 불러오지 못했습니다.');
        } finally {
            setDockerListLoading(false);
        }
    };

    const fetchSettings = async () => {
        try {
            const res = await fetch('/api/settings');
            if (!res.ok) {
                throw new Error('설정을 불러오는 중 오류가 발생했습니다.');
            }
            const settings = await res.json();
            dockerToggle.checked = Boolean(settings.docker_enabled);
            dockerUrlInput.value = settings.docker_url || '';
            detectorBaseUrlInput.value = settings.detector_llm_base_url || settings.llm_base_url || '';
            detectorModelInput.value = settings.detector_llm_model || 'datalab-to/chandra-ocr-2';
            recognizerBaseUrlInput.value = settings.recognizer_llm_base_url || settings.llm_base_url || '';
            recognizerModelInput.value = settings.recognizer_llm_model || 'PaddlePaddle/PaddleOCR-VL-1.5';
            llmApiKeyInput.value = settings.llm_api_key || '';
            kafkaToggle.checked = Boolean(settings.kafka_enabled);
            kafkaUrlInput.value = settings.kafka_url || '';
            toggleDockerFields(dockerToggle.checked);
            toggleKafkaUrl(kafkaToggle.checked);
            await fetchDockerContainers({
                detector: settings.detector_docker_name || '',
                recognizer: settings.recognizer_docker_name || '',
            });
        } catch (error) {
            showFeedback('danger', error.message || '설정을 불러올 수 없습니다.');
        }
    };

    const collectPayload = () => {
        const payload = {
            docker_enabled: dockerToggle.checked,
            docker_url: dockerUrlInput.value.trim(),
            detector_docker_name: detectorDockerNameSelect.value.trim(),
            recognizer_docker_name: recognizerDockerNameSelect.value.trim(),
            detector_llm_model: detectorModelInput.value.trim(),
            recognizer_llm_model: recognizerModelInput.value.trim(),
            kafka_enabled: kafkaToggle.checked,
            kafka_url: kafkaUrlInput.value.trim(),
        };

        const detectorBaseUrl = detectorBaseUrlInput.value.trim();
        const recognizerBaseUrl = recognizerBaseUrlInput.value.trim();
        payload.detector_llm_base_url = detectorBaseUrl.length ? detectorBaseUrl : null;
        payload.recognizer_llm_base_url = recognizerBaseUrl.length ? recognizerBaseUrl : null;
        const llmApiKey = llmApiKeyInput.value.trim();
        payload.llm_api_key = llmApiKey.length ? llmApiKey : null;

        if (!payload.kafka_enabled && !payload.kafka_url.length) {
            // 비활성 상태에서 빈 값이면 서버의 기존 설정을 유지합니다.
            delete payload.kafka_url;
        }

        return payload;
    };

    const validatePayload = (payload) => {
        if (payload.docker_enabled && !payload.docker_url) {
            return 'Docker 자동 제어를 사용하려면 Docker 엔드포인트를 입력해주세요.';
        }
        if (payload.docker_enabled && !payload.detector_docker_name) {
            return 'Docker 자동 제어를 사용하려면 Detector 컨테이너 이름을 선택해주세요.';
        }
        if (payload.docker_enabled && !payload.recognizer_docker_name) {
            return 'Docker 자동 제어를 사용하려면 Recognizer 컨테이너 이름을 선택해주세요.';
        }
        if (payload.docker_enabled && payload.detector_docker_name === payload.recognizer_docker_name) {
            return 'Detector와 Recognizer는 서로 다른 컨테이너를 선택해주세요.';
        }
        if (!payload.detector_llm_base_url) {
            return 'BBox Detector LLM Base URL을 입력해주세요.';
        }
        if (!payload.detector_llm_model) {
            return 'BBox Detector 모델을 입력해주세요.';
        }
        if (!payload.recognizer_llm_base_url) {
            return 'OCR Recognizer LLM Base URL을 입력해주세요.';
        }
        if (!payload.recognizer_llm_model) {
            return 'OCR Recognizer 모델을 입력해주세요.';
        }
        if (payload.kafka_enabled && !payload.kafka_url) {
            return 'Kafka를 사용하려면 Bootstrap 서버 주소를 입력해주세요.';
        }
        return null;
    };

    const setLoadingState = (isLoading) => {
        saveButton.disabled = isLoading;
        saveButton.textContent = isLoading ? '저장 중...' : '변경 사항 저장';
    };

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        clearFeedback();

        const payload = collectPayload();
        const validationError = validatePayload(payload);
        if (validationError) {
            showFeedback('warning', validationError);
            return;
        }

        setLoadingState(true);
        try {
            const res = await fetch('/api/settings', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const responseBody = await res.json().catch(() => ({}));

            if (!res.ok) {
                const detail = responseBody.detail;
                if (Array.isArray(detail)) {
                    const messages = detail.map((item) => item.msg || JSON.stringify(item)).join('\n');
                    showFeedback('danger', messages);
                } else {
                    showFeedback('danger', detail || '설정을 저장하지 못했습니다.');
                }
                return;
            }

            showFeedback('success', '설정을 저장했습니다.');
            kafkaToggle.checked = Boolean(responseBody.kafka_enabled);
            toggleKafkaUrl(kafkaToggle.checked);
        } catch (error) {
            showFeedback('danger', error.message || '설정을 저장하지 못했습니다.');
        } finally {
            setLoadingState(false);
        }
    });

    dockerToggle.addEventListener('change', async () => {
        toggleDockerFields(dockerToggle.checked);
        await fetchDockerContainers(getSelectedDockerNames());
    });
    kafkaToggle.addEventListener('change', () => {
        toggleKafkaUrl(kafkaToggle.checked);
    });
    refreshDockerListButton.addEventListener('click', async () => {
        await fetchDockerContainers(getSelectedDockerNames());
    });
    dockerUrlInput.addEventListener('blur', async () => {
        await fetchDockerContainers(getSelectedDockerNames());
    });

    fetchSettings();
});
