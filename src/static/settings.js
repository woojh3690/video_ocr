document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('settings-form');
    const dockerUrlInput = document.getElementById('docker-url');
    const dockerNameInput = document.getElementById('docker-name');
    const llmBaseUrlInput = document.getElementById('llm-base-url');
    const llmModelInput = document.getElementById('llm-model');
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

    const fetchSettings = async () => {
        try {
            const res = await fetch('/api/settings');
            if (!res.ok) {
                throw new Error('설정을 불러오는 중 오류가 발생했습니다.');
            }
            const settings = await res.json();
            dockerUrlInput.value = settings.docker_url || '';
            dockerNameInput.value = settings.docker_name || '';
            llmBaseUrlInput.value = settings.llm_base_url || '';
            llmModelInput.value = settings.llm_model || '';
            kafkaToggle.checked = Boolean(settings.kafka_enabled);
            kafkaUrlInput.value = settings.kafka_url || '';
            toggleKafkaUrl(kafkaToggle.checked);
        } catch (error) {
            showFeedback('danger', error.message || '설정을 불러올 수 없습니다.');
        }
    };

    const collectPayload = () => {
        const payload = {
            docker_url: dockerUrlInput.value.trim(),
            docker_name: dockerNameInput.value.trim(),
            llm_model: llmModelInput.value.trim(),
            kafka_enabled: kafkaToggle.checked,
            kafka_url: kafkaUrlInput.value.trim(),
        };

        const llmBaseUrl = llmBaseUrlInput.value.trim();
        payload.llm_base_url = llmBaseUrl.length ? llmBaseUrl : null;

        if (!payload.kafka_enabled && !payload.kafka_url.length) {
            // Keep previous value server-side; remove field if empty while disabled.
            delete payload.kafka_url;
        }

        return payload;
    };

    const validatePayload = (payload) => {
        if (!payload.docker_url) {
            return 'Docker 엔드포인트를 입력해주세요.';
        }
        if (!payload.docker_name) {
            return '컨테이너 이름을 입력해주세요.';
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

    kafkaToggle.addEventListener('change', () => {
        toggleKafkaUrl(kafkaToggle.checked);
    });

    fetchSettings();
});
