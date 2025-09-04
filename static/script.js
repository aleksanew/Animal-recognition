let camStream = null;

async function sendUpload() {
  const f = document.getElementById('file').files[0];
  if (!f) { alert("Choose an image first."); return; }
  const fd = new FormData(); fd.append('file', f);
  const res = await fetch('/predict_upload', { method: 'POST', body: fd });
  const data = await res.json();
  document.getElementById('uploadResult').textContent =
      `Prediction: ${data.label} (score ${data.score.toFixed(3)})`;
  document.getElementById('uploadPreview').src = data.image;
}

async function startCam() {
  if (camStream) return; // already running
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    const video = document.getElementById('video');
    video.srcObject = camStream;
    video.play();

    // Enable/disable buttons
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    document.getElementById('captureBtn').disabled = false;
  } catch (e) {
    camStream = null;
    alert("Could not access camera: " + e.message);
  }
}

function stopCam() {
  const video = document.getElementById('video');
  if (camStream) {
    // Stop all tracks
    camStream.getTracks().forEach(t => t.stop());
    camStream = null;
  }
  // Detach stream from video and pause
  video.pause();
  video.srcObject = null;
  // Optionally clear the frame
  // video.src = ""; // uncomment if iOS Safari holds the last frame

  // Disable capture when no camera
  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled = true;
  document.getElementById('captureBtn').disabled = true;
}

async function capture() {
  if (!camStream) { alert("Start camera first."); return; }
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

  const res = await fetch('/predict_snapshot', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataUrl })
  });
  const data = await res.json();
  document.getElementById('camResult').textContent =
      `Prediction: ${data.label} (score ${data.score.toFixed(3)})`;
  document.getElementById('camPreview').src = data.image;
}

// Clean up if the user navigates away or reloads
window.addEventListener('beforeunload', stopCam);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopCam();
});
