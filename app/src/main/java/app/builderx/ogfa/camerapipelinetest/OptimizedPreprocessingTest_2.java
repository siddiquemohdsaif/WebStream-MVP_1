package app.builderx.ogfa.camerapipelinetest;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class OptimizedPreprocessingTest_2 extends AppCompatActivity {
    private static final String TAG = "OptimizedPreprocess2";
    static { System.loadLibrary("camera_pipeline"); }

    private static final String INPUT_ASSET = "input_pipeline_sample.png";
    private static final String[] DOWNSAMPLE_LABELS = {
            "4x4 -> 2x2", "4x4 -> 3x3", "3x3 -> 2x2", "1x1 -> 1x1"
    };
    private static final int[] DOWNSAMPLE_MODES = {4, 5, 3, 1};
    private static final String[] DOWNSAMPLE_SHADERS = {
            "downsample_4to2.comp.spv",
            "downsample_4to3.comp.spv",
            "downsample_3to2.comp.spv",
            "downsample_1to1.comp.spv"
    };
    private static final String SOBEL_SHADER = "optimized_sobel_compact.comp.spv";
    private static final String MEDIAN_SHADER = "optimized_median_v2_compact.comp.spv";
    private static final int Y_THRESHOLD = 80;
    private static final int CHROMA_THRESHOLD = 40;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private Spinner downsampleMode;
    private ProgressBar progress;
    private TextView status;
    private ImageView preview;

    static native String nativeRunFullPipeline(
            Bitmap bitmap,
            byte[] output,
            int width,
            int height,
            int mode,
            byte[] downsampleShader,
            byte[] sobelShader,
            byte[] medianShader,
            int yThreshold,
            int chromaThreshold);

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (16 * getResources().getDisplayMetrics().density);
        root.setPadding(pad, pad, pad, pad);

        TextView modeLabel = new TextView(this);
        modeLabel.setText("Optimized downsampling mode");

        downsampleMode = new Spinner(this);
        downsampleMode.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, DOWNSAMPLE_LABELS));
        downsampleMode.setSelection(1); // Default: 3x3 -> 2x2

        Button run = new Button(this);
        run.setText("Run optimized preprocessing v2");

        progress = new ProgressBar(this);
        progress.setIndeterminate(true);
        progress.setVisibility(ProgressBar.GONE);

        status = new TextView(this);
        status.setText("Ready: " + INPUT_ASSET);

        preview = new ImageView(this);
        preview.setAdjustViewBounds(true);

        root.addView(modeLabel);
        root.addView(downsampleMode);
        root.addView(run);
        root.addView(progress);
        root.addView(status);
        root.addView(preview);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(root);
        setContentView(scroll);

        run.setOnClickListener(v -> runPipeline());
        runPipeline();
    }

    @Override protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }

    private void runPipeline() {
        final int modeIndex = downsampleMode.getSelectedItemPosition();
        progress.setVisibility(ProgressBar.VISIBLE);
        status.setText("Running optimized v2 DEVICE_LOCAL pipeline...");
        executor.execute(() -> {
            try {
                Result result = executePipeline(modeIndex);
                runOnUiThread(() -> {
                    progress.setVisibility(ProgressBar.GONE);
                    preview.setImageBitmap(result.bitmap);
                    status.setText(result.report);
                    Log.i(TAG, "Final PNG saved at: " + result.outputPath);
                    Toast.makeText(this, "Saved: " + result.outputPath, Toast.LENGTH_LONG).show();
                });
            } catch (Exception error) {
                runOnUiThread(() -> {
                    progress.setVisibility(ProgressBar.GONE);
                    status.setText("Error: " + error.getMessage());
                    Log.e(TAG, "Optimized pipeline failed", error);
                });
            }
        });
    }

    private Result executePipeline(int modeIndex) throws Exception {
        Bitmap input = loadAssetBitmap(INPUT_ASSET);
        int mode = DOWNSAMPLE_MODES[modeIndex];
        int widthMultiple = mode == 5 ? 16 : (mode == 3 ? 12 : (mode == 4 ? 8 : 4));
        int heightMultiple = mode == 5 ? 4 : (mode == 3 ? 3 : (mode == 4 ? 2 : 1));
        int srcWidth = input.getWidth() - input.getWidth() % widthMultiple;
        int srcHeight = input.getHeight() - input.getHeight() % heightMultiple;
        int outWidth = mode == 5 ? srcWidth / 4 * 3
                : (mode == 3 ? srcWidth / 3 * 2 : (mode == 4 ? srcWidth / 2 : srcWidth));
        int outHeight = mode == 5 ? srcHeight / 4 * 3
                : (mode == 3 ? srcHeight / 3 * 2 : (mode == 4 ? srcHeight / 2 : srcHeight));
        int planeBytes = outWidth * outHeight;

        byte[] filtered = new byte[planeBytes * 3];
        String nativeReport = nativeRunFullPipeline(
                input,
                filtered,
                srcWidth,
                srcHeight,
                mode,
                readAsset(DOWNSAMPLE_SHADERS[modeIndex]),
                readAsset(SOBEL_SHADER),
                readAsset(MEDIAN_SHADER),
                Y_THRESHOLD,
                CHROMA_THRESHOLD);
        throwIfNativeError(nativeReport);

        Bitmap finalBitmap = yuv444PlanarToBitmap(filtered, outWidth, outHeight);
        File output = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES),
                "optimized_preprocessing_v2_final.png");
        try (FileOutputStream stream = new FileOutputStream(output)) {
            finalBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
        }

        String report = "Saved optimized v2 final PNG:\n" + output.getAbsolutePath()
                + "\n\nInput: " + input.getWidth() + "x" + input.getHeight()
                + "\nTrimmed: " + srcWidth + "x" + srcHeight
                + "\nOutput: " + outWidth + "x" + outHeight
                + "\nDownsampling: " + DOWNSAMPLE_LABELS[modeIndex]
                + "\n\nNative optimized report:\n" + nativeReport;
        return new Result(finalBitmap, report, output.getAbsolutePath());
    }

    private Bitmap loadAssetBitmap(String name) throws IOException {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        try (InputStream stream = getAssets().open(name)) {
            Bitmap bitmap = BitmapFactory.decodeStream(stream, null, options);
            if (bitmap == null) throw new IOException("Could not decode " + name);
            if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
                bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false);
            }
            return bitmap;
        }
    }

    private byte[] readAsset(String name) throws IOException {
        try (InputStream stream = getAssets().open(name);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16 * 1024];
            int read;
            while ((read = stream.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
            return output.toByteArray();
        }
    }

    private void throwIfNativeError(String report) {
        if (report != null && report.startsWith("Error:")) {
            throw new IllegalStateException(report);
        }
    }

    private Bitmap yuv444PlanarToBitmap(byte[] yuv, int width, int height) {
        int pixelCount = width * height;
        int[] argb = new int[pixelCount];
        for (int i = 0; i < pixelCount; i++) {
            int y = yuv[i] & 0xff;
            int cb = (yuv[pixelCount + i] & 0xff) - 128;
            int cr = (yuv[pixelCount * 2 + i] & 0xff) - 128;
            int r = clamp(Math.round(y + 1.402f * cr));
            int g = clamp(Math.round(y - 0.344136f * cb - 0.714136f * cr));
            int b = clamp(Math.round(y + 1.772f * cb));
            argb[i] = 0xff000000 | (r << 16) | (g << 8) | b;
        }
        return Bitmap.createBitmap(argb, width, height, Bitmap.Config.ARGB_8888);
    }

    private int clamp(int value) {
        return value < 0 ? 0 : Math.min(value, 255);
    }

    private static final class Result {
        final Bitmap bitmap;
        final String report;
        final String outputPath;
        Result(Bitmap bitmap, String report, String outputPath) {
            this.bitmap = bitmap;
            this.report = report;
            this.outputPath = outputPath;
        }
    }
}
