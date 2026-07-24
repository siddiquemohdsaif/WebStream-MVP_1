package app.builderx.ogfa.camerapipelinetest;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class StartCallActivity extends AppCompatActivity implements SurfaceHolder.Callback {
    static { System.loadLibrary("camera_pipeline"); }

    private static final int CAMERA_PERMISSION = 43;
    private static final int CAMERA_WIDTH = 2560;
    private static final int CAMERA_HEIGHT = 1440;

    private SurfaceView surfaceView;
    private SurfaceView floatingSurfaceView;
    private TextView status;
    private View form;
    private View cameraContainer;
    private boolean front;
    private boolean resumed;
    private boolean connected;
    private boolean surfaceReady;
    private final ExecutorService captureExecutor = Executors.newSingleThreadExecutor();

    private static native String nativeStart(boolean front, int width, int height);
    private static native void nativeStop();
    private static native String nativeSetSurface(Surface surface, android.content.res.AssetManager assets);
    private static native void nativeSetMainPreviewRendering(boolean enabled);
    private static native void nativeSetDisplayRotation(int rotationDegrees);
    private static native byte[][] nativeCaptureFilteredYuv420Ring(int[] info);

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.activity_start_call);
        form = findViewById(R.id.startCallForm);
        cameraContainer = findViewById(R.id.startCallCameraContainer);
        status = findViewById(R.id.cameraStatus);
        surfaceView = findViewById(R.id.vulkanSurface);
        floatingSurfaceView = findViewById(R.id.floatingVulkanSurface);
        floatingSurfaceView.setZOrderMediaOverlay(true);
        floatingSurfaceView.getHolder().addCallback(this);

        EditText userId = findViewById(R.id.startCallUserId);
        EditText callId = findViewById(R.id.startCallCallId);
        findViewById(R.id.connectCall).setOnClickListener(v -> {
            connected = true;
            nativeSetMainPreviewRendering(true);
            form.setVisibility(View.GONE);
            cameraContainer.setVisibility(View.VISIBLE);
            status.setText("Connecting " + userId.getText() + " to " + callId.getText() + "...");
            surfaceView.setZOrderOnTop(false);
            updateDisplayRotation();
            requestCameraIfNeeded();
        });
        findViewById(R.id.flipCamera).setOnClickListener(v -> {
            front = !front;
            nativeStop();
            startIfReady();
        });
        findViewById(R.id.saveFilteredFrame).setOnClickListener(v -> saveFilteredFrame());
    }

    @Override protected void onResume() {
        super.onResume();
        resumed = true;
        updateDisplayRotation();
        if (connected) requestCameraIfNeeded();
    }

    @Override protected void onPause() {
        resumed = false;
        nativeStop();
        nativeSetMainPreviewRendering(true);
        super.onPause();
    }

    @Override protected void onDestroy() {
        nativeSetMainPreviewRendering(true);
        captureExecutor.shutdown();
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

    @Override public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                                      @NonNull int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode == CAMERA_PERMISSION && results.length > 0
                && results[0] == PackageManager.PERMISSION_GRANTED) startIfReady();
        else if (requestCode == CAMERA_PERMISSION) status.setText("Camera permission denied");
    }
}
