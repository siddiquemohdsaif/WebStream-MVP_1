package app.builderx.ogfa.camerapipelinetest;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Locale;
import java.util.Queue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class StartCallActivity extends AppCompatActivity implements SurfaceHolder.Callback {
    static { System.loadLibrary("camera_pipeline"); }

    private static final String TAG = "XX_PIPELINE_CALL";
    private static final int CAMERA_PERMISSION = 43;
    private static final int CAMERA_WIDTH = 2560;
    private static final int CAMERA_HEIGHT = 1440;
    private static final int maxFPS = 30;
    private static final int DEFAULT_JPEG_QUALITY_PERCENT = 82;

    private SurfaceView surfaceView;
    private SurfaceView floatingSurfaceView;
    private TextView status;
    private TextView connectionStatus;
    private TextView callTimerText;
    private EditText userId;
    private EditText callId;
    private EditText jpegQualityPercent;
    private Button connectButton;
    private View form;
    private View cameraContainer;
    private boolean front;
    private boolean resumed;
    private boolean connected;
    private boolean surfaceReady;
    private boolean remoteSurfaceReady;
    private boolean receivedRenderRunning;
    private long callStartedAtMs;
    private volatile int currentJpegQualityPercent = DEFAULT_JPEG_QUALITY_PERCENT;
    private final Object receivedRenderLock = new Object();
    private final Queue<byte[]> receivedJpegQueue = new ArrayDeque<>();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final ExecutorService captureExecutor = Executors.newSingleThreadExecutor();
    private final ExecutorService receivedRenderExecutor = Executors.newSingleThreadExecutor();
    private LatestFilteredFrameWorker latestFilteredFrameWorker;
    private WebStreamClient webStreamClient;
    private WebStreamCall webStreamCall;
    private final CallJpegStats callJpegStats = new CallJpegStats();
    private final Runnable timerTick = new Runnable() {
        @Override public void run() {
            if (!connected || callStartedAtMs == 0L) return;
            long elapsedSeconds = Math.max(0L, (System.currentTimeMillis() - callStartedAtMs) / 1000L);
            callTimerText.setText(formatDuration(elapsedSeconds));
            mainHandler.postDelayed(this, 1000L);
        }
    };

    private static native String nativeStart(boolean front, int width, int height);
    private static native void nativeStop();
    private static native String nativeSetSurface(Surface surface, android.content.res.AssetManager assets);
    private static native String nativeSetRemoteSurface(Surface surface);
    private static native void nativeSetMainPreviewRendering(boolean enabled);
    private static native void nativeSetDisplayRotation(int rotationDegrees);
    private static native boolean nativeRenderJpegToMainSurface(byte[] jpeg, int rotationDegrees, boolean mirror);
    private static native byte[][] nativeCaptureFilteredYuv420Ring(int[] info);

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.activity_start_call);
        form = findViewById(R.id.startCallForm);
        cameraContainer = findViewById(R.id.startCallCameraContainer);
        status = findViewById(R.id.cameraStatus);
        connectionStatus = findViewById(R.id.callConnectionStatus);
        callTimerText = findViewById(R.id.callTimerText);
        userId = findViewById(R.id.startCallUserId);
        callId = findViewById(R.id.startCallCallId);
        jpegQualityPercent = findViewById(R.id.startCallJpegQualityPercent);
        connectButton = findViewById(R.id.connectCall);
        surfaceView = findViewById(R.id.vulkanSurface);
        floatingSurfaceView = findViewById(R.id.floatingVulkanSurface);
        surfaceView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override public void surfaceCreated(@NonNull SurfaceHolder holder) {
                String result = nativeSetRemoteSurface(holder.getSurface());
                remoteSurfaceReady = !result.startsWith("Error:");
                status.setText(result);
            }

            @Override public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int width, int height) {
                String result = nativeSetRemoteSurface(holder.getSurface());
                remoteSurfaceReady = !result.startsWith("Error:");
                status.setText(result);
            }

            @Override public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
                remoteSurfaceReady = false;
                nativeSetRemoteSurface(null);
            }
        });
        floatingSurfaceView.setZOrderMediaOverlay(true);
        floatingSurfaceView.getHolder().addCallback(this);

        latestFilteredFrameWorker = new LatestFilteredFrameWorker(this,
                new LatestFilteredFrameWorker.FilteredFrameSource() {
                    @Override public int getMaxFps() {
                        return maxFPS;
                    }

                    @Override public int getJpegQualityPercent() {
                        return currentJpegQualityPercent;
                    }

                    @Override public LatestFilteredFrameWorker.SendResult onLatestFilteredJpeg(
                            byte[] jpeg, int width, int height, long timestampMs) {
                        WebStreamCall activeCall = webStreamCall;
                        long startNs = System.nanoTime();
                        boolean sent = false;
                        boolean queued = false;
                        long sequence = 0L;
                        if (activeCall != null && activeCall.getState() == WebStreamCall.State.CONNECTED) {
                            WebStreamCall.SendOutcome outcome =
                                    activeCall.sendJpeg(jpeg, width, height, maxFPS, 0, timestampMs);
                            sent = outcome.sent;
                            queued = outcome.queued;
                            sequence = outcome.sequence;
                            if (sent) callJpegStats.recordSent();
                        }
                        return new LatestFilteredFrameWorker.SendResult(
                                sent,
                                queued,
                                sequence,
                                (System.nanoTime() - startNs) / 1_000_000.0);
                    }
                });

        connectButton.setOnClickListener(v -> connectCall());
        findViewById(R.id.flipCamera).setOnClickListener(v -> {
            front = !front;
            nativeStop();
            startIfReady();
        });
        findViewById(R.id.saveFilteredFrame).setOnClickListener(v -> saveFilteredFrame());
        findViewById(R.id.disconnectCall).setOnClickListener(v -> disconnectCall());
    }

    @Override protected void onResume() {
        super.onResume();
        resumed = true;
        updateDisplayRotation();
        if (connected) {
            latestFilteredFrameWorker.start();
            requestCameraIfNeeded();
        }
    }

    @Override protected void onPause() {
        resumed = false;
        latestFilteredFrameWorker.stop();
        nativeStop();
        nativeSetMainPreviewRendering(true);
        super.onPause();
    }

    @Override protected void onDestroy() {
        nativeSetMainPreviewRendering(true);
        mainHandler.removeCallbacks(timerTick);
        clearReceivedRenderQueue();
        leaveCall();
        latestFilteredFrameWorker.shutdown();
        captureExecutor.shutdown();
        receivedRenderExecutor.shutdown();
        super.onDestroy();
    }

    @Override public void surfaceCreated(@NonNull SurfaceHolder holder) {
        String result = nativeSetSurface(holder.getSurface(), getAssets());
        nativeSetMainPreviewRendering(true);
        surfaceReady = !result.startsWith("Error:");
        status.setText(result);
        startIfReady();
    }

    @Override public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int width, int height) {
        updateDisplayRotation();
        nativeStop();
        String result = nativeSetSurface(holder.getSurface(), getAssets());
        nativeSetMainPreviewRendering(true);
        surfaceReady = !result.startsWith("Error:");
        status.setText(result);
        startIfReady();
    }

    @Override public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
        surfaceReady = false;
        nativeStop();
        nativeSetSurface(null, getAssets());
    }

    private void connectCall() {
        String currentUserId = userId.getText().toString().trim();
        String currentCallId = callId.getText().toString().trim();
        if (currentUserId.isEmpty()) {
            userId.setError("Required");
            return;
        }
        if (currentCallId.isEmpty()) {
            callId.setError("Required");
            return;
        }
        currentJpegQualityPercent = readJpegQualityPercentFromInput();
        jpegQualityPercent.setText(String.valueOf(currentJpegQualityPercent));
        connectButton.setEnabled(false);
        Log.d(TAG, "connect_click | t=" + System.currentTimeMillis()
                + " userId=" + currentUserId
                + " callId=" + currentCallId
                + " jpegQualityPercent=" + currentJpegQualityPercent);
        connectionStatus.setText(String.format(Locale.US, "Connecting %s to %s...",
                currentUserId, currentCallId));
        initializeCall(currentUserId, currentCallId);
    }

    private void initializeCall(String currentUserId, String currentCallId) {
        leaveCall();
        webStreamClient = new WebStreamClient.Builder(this)
                .userId(currentUserId)
                .displayName(currentUserId)
                .build();
        webStreamCall = webStreamClient.joinCall(currentCallId, new WebStreamClient.Listener() {
            @Override public void onConnecting() {
                connectionStatus.setText("Connecting devices...");
            }

            @Override public void onWaitingForPeer() {
                connectionStatus.setText("Waiting for peer...");
            }

            @Override public void onConnected() {
                long startNs = System.nanoTime();
                connected = true;
                nativeSetMainPreviewRendering(true);
                form.setVisibility(View.GONE);
                cameraContainer.setVisibility(View.VISIBLE);
                surfaceView.setZOrderOnTop(false);
                callStartedAtMs = System.currentTimeMillis();
                callJpegStats.start(callStartedAtMs);
                callTimerText.setText("00:00");
                mainHandler.removeCallbacks(timerTick);
                mainHandler.post(timerTick);
                latestFilteredFrameWorker.start();
                updateDisplayRotation();
                requestCameraIfNeeded();
                status.setText("Call connected");
                Log.d(TAG, "call_connected_ui_ready | t=" + System.currentTimeMillis()
                        + " elapsedMs=" + ((System.nanoTime() - startNs) / 1_000_000.0));
            }

            @Override public void onJpegReceived(WebStreamJpegFrame frame) {
                callJpegStats.recordReceived();
                renderReceivedJpegAsync(frame.getJpegData());
                connectionStatus.setText(String.format(Locale.US,
                        "Received JPEG %dx%d seq=%d",
                        frame.getWidth(), frame.getHeight(), frame.getSequence()));
            }

            @Override public void onDisconnected() {
                showPreCallUi(callJpegStats.summary("Call disconnected"));
            }

            @Override public void onError(Throwable error) {
                String reason = "Call error: " + (error == null ? "unknown" : error.getMessage());
                showPreCallUi(callJpegStats.summary(reason));
            }
        });
    }

    private void showPreCallUi(String message) {
        connected = false;
        callStartedAtMs = 0L;
        callJpegStats.reset();
        clearReceivedRenderQueue();
        mainHandler.removeCallbacks(timerTick);
        callTimerText.setText("00:00");
        latestFilteredFrameWorker.stop();
        nativeStop();
        cameraContainer.setVisibility(View.GONE);
        form.setVisibility(View.VISIBLE);
        connectButton.setEnabled(true);
        connectionStatus.setText(message == null ? "" : message);
    }

    private void leaveCall() {
        if (webStreamCall != null) {
            webStreamCall.leave();
            webStreamCall = null;
        }
        if (webStreamClient != null) {
            webStreamClient.release();
            webStreamClient = null;
        }
    }

    private void disconnectCall() {
        String summary = callJpegStats.summary("Call disconnected");
        leaveCall();
        showPreCallUi(summary);
    }

    private void renderReceivedJpegAsync(byte[] jpegData) {
        if (jpegData == null || jpegData.length == 0) return;
        synchronized (receivedRenderLock) {
            receivedJpegQueue.add(jpegData);
            if (receivedRenderRunning) {
                Log.d(TAG, "remote_render_frame_queued | t=" + System.currentTimeMillis()
                        + " bytes=" + jpegData.length
                        + " queueDepth=" + receivedJpegQueue.size());
                return;
            }
            receivedRenderRunning = true;
        }

        receivedRenderExecutor.execute(() -> {
            while (true) {
                byte[] frame;
                synchronized (receivedRenderLock) {
                    frame = receivedJpegQueue.poll();
                    if (frame == null) {
                        receivedRenderRunning = false;
                        return;
                    }
                }

                boolean rendered = remoteSurfaceReady
                        && nativeRenderJpegToMainSurface(frame, 0, false);
                if (!rendered) {
                    runOnUiThread(() -> status.setText("Received JPEG render skipped"));
                }
            }
        });
    }

    private void clearReceivedRenderQueue() {
        synchronized (receivedRenderLock) {
            receivedJpegQueue.clear();
            receivedRenderRunning = false;
        }
    }

    private void requestCameraIfNeeded() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION);
        } else {
            startIfReady();
        }
    }

    private void startIfReady() {
        if (!connected || !resumed || !surfaceReady || ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) return;
        nativeSetMainPreviewRendering(true);
        status.setText(nativeStart(front, CAMERA_WIDTH, CAMERA_HEIGHT));
    }

    private void updateDisplayRotation() {
        if (getDisplay() != null) nativeSetDisplayRotation(getDisplay().getRotation() * 90);
    }

    private void saveFilteredFrame() {
        status.setText("Pausing preview to clone filtered YUV ring...");
        captureExecutor.execute(() -> {
            String message;
            try {
                int[] info = new int[9];
                byte[][] frames = nativeCaptureFilteredYuv420Ring(info);
                if (frames == null || frames.length != 6 || info[0] <= 0 || info[1] <= 0) {
                    message = "Need three filtered frames before saving";
                } else {
                    File dir = getExternalFilesDir(Environment.DIRECTORY_MOVIES);
                    if (dir == null) dir = getFilesDir();
                    if (!dir.exists() && !dir.mkdirs()) {
                        throw new IOException("Failed to create " + dir.getAbsolutePath());
                    }
                    long timestamp = System.currentTimeMillis();
                    for (int i = 0; i < 6; ++i) {
                        String stage = i < 3 ? "downsampled" : "filtered";
                        int ringIndex = i < 3 ? i : i - 3;
                        File file = new File(dir, String.format(Locale.US,
                                "%s_%d_n_%d_y%dx%d_uv%dx%d.yuv",
                                stage, timestamp, ringIndex,
                                info[0], info[1], info[2], info[3]));
                        try (FileOutputStream out = new FileOutputStream(file)) {
                            writeUnpaddedYuv420(out, frames[i], info);
                        }
                    }
                    message = "Saved 6 unpadded YUV frames: downsampled_n_0..2 and filtered_n_0..2 in "
                            + dir.getAbsolutePath();
                }
            } catch (Exception e) {
                message = "Error saving filtered YUV ring: " + e.getMessage();
            }
            String finalMessage = message;
            runOnUiThread(() -> status.setText(finalMessage));
        });
    }

    private static void writeUnpaddedYuv420(FileOutputStream out, byte[] paddedFrame, int[] info)
            throws IOException {
        int width = info[0];
        int height = info[1];
        int chromaWidth = info[2];
        int chromaHeight = info[3];
        int paddedWidth = info[4];
        int paddedHeight = info[5];
        int paddedChromaWidth = info[6];
        int paddedChromaHeight = info[7];
        int paddedYSize = paddedWidth * paddedHeight;
        int paddedChromaSize = paddedChromaWidth * paddedChromaHeight;
        int expectedPaddedSize = paddedYSize + 2 * paddedChromaSize;
        if (paddedFrame.length < expectedPaddedSize) {
            throw new IOException("YUV frame too small: " + paddedFrame.length
                    + " < " + expectedPaddedSize);
        }

        for (int row = 0; row < height; ++row) {
            out.write(paddedFrame, row * paddedWidth, width);
        }
        int uOffset = paddedYSize;
        int vOffset = paddedYSize + paddedChromaSize;
        for (int row = 0; row < chromaHeight; ++row) {
            out.write(paddedFrame, uOffset + row * paddedChromaWidth, chromaWidth);
        }
        for (int row = 0; row < chromaHeight; ++row) {
            out.write(paddedFrame, vOffset + row * paddedChromaWidth, chromaWidth);
        }
    }

    private String formatDuration(long elapsedSeconds) {
        long minutes = elapsedSeconds / 60L;
        long seconds = elapsedSeconds % 60L;
        return String.format(Locale.US, "%02d:%02d", minutes, seconds);
    }

    private int readJpegQualityPercentFromInput() {
        if (jpegQualityPercent == null) return DEFAULT_JPEG_QUALITY_PERCENT;
        String value = jpegQualityPercent.getText().toString().trim();
        if (value.isEmpty()) return DEFAULT_JPEG_QUALITY_PERCENT;
        try {
            int percent = Integer.parseInt(value);
            return Math.max(1, Math.min(100, percent));
        } catch (NumberFormatException ignored) {
            return DEFAULT_JPEG_QUALITY_PERCENT;
        }
    }

    private static final class CallJpegStats {
        private long connectedAtMs;
        private long disconnectedAtMs;
        private int sentJpegs;
        private int receivedJpegs;

        synchronized void start(long nowMs) {
            connectedAtMs = nowMs;
            disconnectedAtMs = 0L;
            sentJpegs = 0;
            receivedJpegs = 0;
        }

        synchronized void recordSent() {
            sentJpegs++;
        }

        synchronized void recordReceived() {
            receivedJpegs++;
        }

        synchronized String summary(String title) {
            if (connectedAtMs == 0L) {
                return title == null ? "" : title;
            }
            disconnectedAtMs = System.currentTimeMillis();
            double connectedSeconds = Math.max(0.001,
                    (disconnectedAtMs - connectedAtMs) / 1000.0);
            return String.format(Locale.US,
                    "%s\nConnected seconds: %.1f\nSent frames: %d (%.2f fps)\nReceived frames: %d (%.2f fps)",
                    title == null ? "Call summary" : title,
                    connectedSeconds,
                    sentJpegs,
                    sentJpegs / connectedSeconds,
                    receivedJpegs,
                    receivedJpegs / connectedSeconds);
        }

        synchronized void reset() {
            connectedAtMs = 0L;
            disconnectedAtMs = 0L;
            sentJpegs = 0;
            receivedJpegs = 0;
        }
    }

    @Override public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                                      @NonNull int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode == CAMERA_PERMISSION && results.length > 0
                && results[0] == PackageManager.PERMISSION_GRANTED) startIfReady();
        else if (requestCode == CAMERA_PERMISSION) status.setText("Camera permission denied");
    }
}
