// Frontend logic for squat form video upload and results display

document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('dropZone');
  const dropZonePrompt = document.getElementById('dropZonePrompt');
  const videoFileInput = document.getElementById('videoFileInput');
  const videoPreviewWrapper = document.getElementById('videoPreviewWrapper');
  const videoPreview = document.getElementById('videoPreview');
  const videoFileName = document.getElementById('videoFileName');
  const changeVideoBtn = document.getElementById('changeVideoBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const errorBanner = document.getElementById('errorBanner');
  const errorMessage = document.getElementById('errorMessage');

  const resultIdleState = document.getElementById('resultIdleState');
  const loadingState = document.getElementById('loadingState');
  const loadingStepTitle = document.getElementById('loadingStepTitle');
  const loadingDesc = document.getElementById('loadingDesc');
  const resultActiveState = document.getElementById('resultActiveState');

  const verdictBanner = document.getElementById('verdictBanner');
  const verdictIcon = document.getElementById('verdictIcon');
  const verdictTitle = document.getElementById('verdictTitle');
  const confidenceValue = document.getElementById('confidenceValue');
  const goodProbVal = document.getElementById('goodProbVal');
  const badProbVal = document.getElementById('badProbVal');
  const goodProgressBar = document.getElementById('goodProgressBar');
  const badProgressBar = document.getElementById('badProgressBar');
  const statModel = document.getElementById('statModel');
  const statFrames = document.getElementById('statFrames');
  const resetBtn = document.getElementById('resetBtn');
  const modelStatusBadge = document.getElementById('modelStatusBadge');

  let currentFile = null;
  let previewUrl = null;
  let loadingInterval = null;

  // Check API status
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        const modelName = data.model?.name || 'GRU';
        const frames = data.model?.frames || 32;
        modelStatusBadge.textContent = `${modelName} Active (${frames}f)`;
      }
    } catch {
      modelStatusBadge.textContent = 'API Initializing...';
    }
  }
  checkHealth();

  // Drag and drop handlers
  ['dragenter', 'dragover'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      handleSelectedFile(files[0]);
    }
  });

  videoFileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleSelectedFile(files[0]);
    }
  });

  function handleSelectedFile(file) {
    hideError();

    // Validate type
    const validExts = ['.mp4', '.mov', '.avi', '.webm', '.mkv'];
    const hasValidExt = validExts.some(ext => file.name.toLowerCase().endsWith(ext));
    if (!file.type.startsWith('video/') && !hasValidExt) {
      showError('Please select a valid video file (.mp4, .mov, .avi, .webm).');
      return;
    }

    // Validate size (100MB)
    if (file.size > 100 * 1024 * 1024) {
      showError('File is too large. Please select a video smaller than 100 MB.');
      return;
    }

    currentFile = file;
    videoFileName.textContent = file.name;

    // Create object URL for preview
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    previewUrl = URL.createObjectURL(file);
    videoPreview.src = previewUrl;

    // Toggle UI views
    dropZone.classList.add('hidden');
    videoPreviewWrapper.classList.remove('hidden');
    analyzeBtn.disabled = false;
  }

  // Change video button
  changeVideoBtn.addEventListener('click', resetSelection);

  function resetSelection() {
    currentFile = null;
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    videoPreview.src = '';
    videoFileInput.value = '';
    dropZone.classList.remove('hidden');
    videoPreviewWrapper.classList.add('hidden');
    analyzeBtn.disabled = true;
    hideError();
  }

  // Run analysis
  analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    hideError();
    setLoadingState(true);

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Server responded with status ${response.status}`);
      }

      renderResults(data);
    } catch (err) {
      showError(err.message || 'An unexpected error occurred during analysis.');
      setLoadingState(false, true);
    } finally {
      setLoadingState(false);
    }
  });

  function setLoadingState(isLoading, revertToIdle = false) {
    if (isLoading) {
      resultIdleState.classList.add('hidden');
      resultActiveState.classList.add('hidden');
      loadingState.classList.remove('hidden');
      analyzeBtn.disabled = true;

      const steps = [
        { title: 'Sampling Video Frames', desc: 'Sampling frames across video duration...' },
        { title: 'Detecting Joint Coordinates', desc: 'Running YOLOv8-pose keypoint estimation...' },
        { title: 'Biomechanical Feature Extraction', desc: 'Calculating joint angles and kinematics...' },
        { title: 'Model Classification', desc: 'Predicting form quality with ensemble model...' },
      ];
      let stepIdx = 0;
      loadingInterval = setInterval(() => {
        stepIdx = (stepIdx + 1) % steps.length;
        loadingStepTitle.textContent = steps[stepIdx].title;
        loadingDesc.textContent = steps[stepIdx].desc;
      }, 1400);

    } else {
      clearInterval(loadingInterval);
      loadingState.classList.add('hidden');
      analyzeBtn.disabled = !currentFile;

      if (revertToIdle) {
        resultIdleState.classList.remove('hidden');
        resultActiveState.classList.add('hidden');
      }
    }
  }

  // Render results
  function renderResults(data) {
    resultIdleState.classList.add('hidden');
    loadingState.classList.add('hidden');
    resultActiveState.classList.remove('hidden');

    const isGood = data.is_good;
    verdictBanner.className = `verdict-banner ${isGood ? 'is-good' : 'is-bad'}`;

    // Verdict Title & Tag
    verdictTitle.textContent = isGood ? 'Good Form' : 'Needs Correction';

    // SVG Icon
    if (isGood) {
      verdictIcon.innerHTML = `<polyline points="20 6 9 17 4 12"></polyline>`;
    } else {
      verdictIcon.innerHTML = `<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>`;
    }

    // Confidence
    const confPct = data.confidence_percent || `${(data.confidence * 100).toFixed(1)}%`;
    confidenceValue.textContent = confPct;

    // Probability Bars
    const goodPct = (data.probabilities.good * 100).toFixed(1);
    const badPct = (data.probabilities.bad * 100).toFixed(1);

    goodProbVal.textContent = `${goodPct}%`;
    badProbVal.textContent = `${badPct}%`;

    // Trigger smooth transition
    goodProgressBar.style.width = '0%';
    badProgressBar.style.width = '0%';
    setTimeout(() => {
      goodProgressBar.style.width = `${goodPct}%`;
      badProgressBar.style.width = `${badPct}%`;
    }, 50);

    // Diagnostics stats
    statModel.textContent = data.model || 'GRU';
    statFrames.textContent = `${data.frames_analyzed || 32} Frames`;
  }

  // Reset Button
  resetBtn.addEventListener('click', () => {
    resetSelection();
    resultActiveState.classList.add('hidden');
    resultIdleState.classList.remove('hidden');
  });

  // Helpers
  function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.classList.remove('hidden');
  }

  function hideError() {
    errorBanner.classList.add('hidden');
    errorMessage.textContent = '';
  }
});
