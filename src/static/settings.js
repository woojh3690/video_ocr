document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('settings-form');
    const dockerToggle = document.getElementById('docker-enabled');
    const dockerFields = document.getElementById('docker-fields');
    const dockerUrlInput = document.getElementById('docker-url');
    const dockerNameSelect = document.getElementById('docker-name');
    const refreshDockerListButton = document.getElementById('refresh-docker-list');
    const llmBaseUrlInput = document.getElementById('llm-base-url');
    const llmModelInput = document.getElementById('llm-model');
    const llmApiKeyInput = document.getElementById('llm-api-key');
    const kafkaToggle = document.getElementById('kafka-enabled');
    const kafkaUrlInput = document.getElementById('kafka-url');
    const feedback = document.getElementById('settings-feedback');
    const saveButton = document.getElementById('save-button');

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
        dockerUrlInput.disabled = !enabled;
        dockerNameSelect.disabled = !enabled;
        refreshDockerListButton.disabled = !enabled;
        dockerFields.classList.toggle('disabled-field', !enabled);
    };

    const setDockerListLoading = (isLoading) => {
        dockerNameSelect.disabled = isLoading;
        refreshDockerListButton.disabled = isLoading;
    };

    const renderDockerOptions = (containers, selectedName) => {
        dockerNameSelect.innerHTML = '';

        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = '컨테이너를 선택하세요'
        dockerNameSelect.appendChild(placeholder);

        const knownNames = new Set();
        containers.forEach((containerName) => {
            const option = document.createElement('option');
            option.value = containerName;
            option.textContent = containerName;
            dockerNameSelect.appendChild(option);
            knownNames.add(containerName);
        });

        if (selectedName && !knownNames.has(selectedName)) {
            const missingOption = document.createElement('option');
            missingOption.value = selectedName;
            missingOption.textContent = `${selectedName} (현재 설정 - 목록에 없음)`;
            dockerNameSelect.appendChild(missingOption);
        }

        if (selectedName) {
            dockerNameSelect.value = selectedName;
        }
    };

    const fetchDockerContainers = async (selectedName) => {
        if (!dockerToggle.checked) {
            renderDockerOptions([], selectedName);
            return;
        }

        const dockerUrl = dockerUrlInput.value.trim();
        if (!dockerUrl) {
            renderDockerOptions([], selectedName);
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
            renderDockerOptions(containers, selectedName);
        } catch (error) {
            renderDockerOptions([], selectedName);
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
            llmBaseUrlInput.value = settings.llm_base_url || '';
            llmModelInput.value = settings.llm_model || '';
            llmApiKeyInput.value = settings.llm_api_key || '';
            kafkaToggle.checked = Boolean(settings.kafka_enabled);
            kafkaUrlInput.value = settings.kafka_url || '';
            toggleDockerFields(dockerToggle.checked);
            toggleKafkaUrl(kafkaToggle.checked);
            await fetchDockerContainers(settings.docker_name || '');
        } catch (error) {
            showFeedback('danger', error.message || '설정을 불러올 수 없습니다.');
        }
    };

    const collectPayload = () => {
        const payload = {
            docker_enabled: dockerToggle.checked,
            docker_url: dockerUrlInput.value.trim(),
            docker_name: dockerNameSelect.value.trim(),
            llm_model: llmModelInput.value.trim(),
            kafka_enabled: kafkaToggle.checked,
            kafka_url: kafkaUrlInput.value.trim(),
        };

        const llmBaseUrl = llmBaseUrlInput.value.trim();
        payload.llm_base_url = llmBaseUrl.length ? llmBaseUrl : null;
        const llmApiKey = llmApiKeyInput.value.trim();
        payload.llm_api_key = llmApiKey.length ? llmApiKey : null;

        if (!payload.kafka_enabled && !payload.kafka_url.length) {
            // Keep previous value server-side; remove field if empty while disabled.
            delete payload.kafka_url;
        }

        return payload;
    };

    const validatePayload = (payload) => {
        if (payload.docker_enabled && !payload.docker_url) {
            return 'Docker 자동 제어를 사용하려면 Docker 엔드포인트를 입력해주세요.';
        }
        if (payload.docker_enabled && !payload.docker_name) {
            return 'Docker 자동 제어를 사용하려면 컨테이너 이름을 선택해주세요.';
        }
        if (!payload.llm_model) {
            return 'LLM 모델을 입력해주세요.';
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
        await fetchDockerContainers(dockerNameSelect.value.trim());
    });
    kafkaToggle.addEventListener('change', () => {
        toggleKafkaUrl(kafkaToggle.checked);
    });
    refreshDockerListButton.addEventListener('click', async () => {
        await fetchDockerContainers(dockerNameSelect.value.trim());
    });
    dockerUrlInput.addEventListener('blur', async () => {
        await fetchDockerContainers(dockerNameSelect.value.trim());
    });

    fetchSettings();
});
