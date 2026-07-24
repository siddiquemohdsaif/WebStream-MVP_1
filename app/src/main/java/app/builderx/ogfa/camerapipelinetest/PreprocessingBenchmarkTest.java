package app.builderx.ogfa.camerapipelinetest;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PreprocessingBenchmarkTest extends AppCompatActivity {
    private static final String TAG = "PreprocessBenchmark";
    static { System.loadLibrary("camera_pipeline"); }

    private static final String INPUT_ASSET = "input_pipeline_sample.png";
    private static final int RUNS = 100;
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

    private static final Pattern TIME_PATTERN =
            Pattern.compile("^\\s*(.+?):\\s*([0-9]+(?:\\.[0-9]+)?)\\s*ms.*$");

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private Spinner downsampleMode;
    private ProgressBar progress;
    private TextView status;
    private Button runButton;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (16 * getResources().getDisplayMetrics().density);
        root.setPadding(pad, pad, pad, pad);

        TextView modeLabel = new TextView(this);
        modeLabel.setText("Benchmark downsampling mode");

        downsampleMode = new Spinner(this);
        downsampleMode.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, DOWNSAMPLE_LABELS));
        downsampleMode.setSelection(1);

        runButton = new Button(this);
        runButton.setText("Run 100x benchmark: V1 vs V2");

        progress = new ProgressBar(this);
        progress.setIndeterminate(true);
        progress.setVisibility(ProgressBar.GONE);

        status = new TextView(this);
        status.setText("Ready. Runs per version: " + RUNS);

        root.addView(modeLabel);
        root.addView(downsampleMode);
        root.addView(runButton);
        root.addView(progress);
        root.addView(status);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(root);
        setContentView(scroll);

        runButton.setOnClickListener(v -> runBenchmark());
    }

    @Override protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }

    private void runBenchmark() {
        final int modeIndex = downsampleMode.getSelectedItemPosition();
        progress.setVisibility(ProgressBar.VISIBLE);
        runButton.setEnabled(false);
        status.setText("Benchmark running...");
        executor.execute(() -> {
            try {
                String report = executeBenchmark(modeIndex);
                runOnUiThread(() -> {
                    progress.setVisibility(ProgressBar.GONE);
                    runButton.setEnabled(true);
                    status.setText(report);
                });
            } catch (Exception error) {
                runOnUiThread(() -> {
                    progress.setVisibility(ProgressBar.GONE);
                    runButton.setEnabled(true);
                    status.setText("Error: " + error.getMessage());
                });
            }
        });
    }

    private String executeBenchmark(int modeIndex) throws Exception {
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
        byte[] output = new byte[outWidth * outHeight * 3];

        byte[] downsampleShader = readAsset(DOWNSAMPLE_SHADERS[modeIndex]);
        byte[] sobelShader = readAsset(SOBEL_SHADER);
        byte[] medianShader = readAsset(MEDIAN_SHADER);

        Stats v1 = benchmarkVersion("V1 shared/coherent", false, input, output,
                srcWidth, srcHeight, mode, downsampleShader, sobelShader, medianShader);
        Stats v2 = benchmarkVersion("V2 device-local", true, input, output,
                srcWidth, srcHeight, mode, downsampleShader, sobelShader, medianShader);

        String report = buildReport(modeIndex, input, srcWidth, srcHeight, outWidth, outHeight, v1, v2);
        Log.i(TAG, report);
        return report;
    }

    private Stats benchmarkVersion(String name,
                                   boolean v2,
                                   Bitmap input,
                                   byte[] output,
                                   int srcWidth,
                                   int srcHeight,
                                   int mode,
                                   byte[] downsampleShader,
                                   byte[] sobelShader,
                                   byte[] medianShader) {
        Stats stats = new Stats(name);
        for (int i = 0; i < RUNS; i++) {
            final long wallStart = SystemClock.elapsedRealtimeNanos();
            String report = v2
                    ? OptimizedPreprocessingTest_2.nativeRunFullPipeline(
                    input, output, srcWidth, srcHeight, mode,
                    downsampleShader, sobelShader, medianShader,
                    Y_THRESHOLD, CHROMA_THRESHOLD)
                    : OptimizedPreprocessingTest.nativeRunFullPipeline(
                    input, output, srcWidth, srcHeight, mode,
                    downsampleShader, sobelShader, medianShader,
                    Y_THRESHOLD, CHROMA_THRESHOLD);
            final double wallMs = (SystemClock.elapsedRealtimeNanos() - wallStart) / 1_000_000.0;
            if (report != null && report.startsWith("Error:")) {
                throw new IllegalStateException(name + " failed at run " + (i + 1) + ": " + report);
            }
            stats.add("Java wall total", wallMs);
            parseNativeTimings(report, stats);
            final int completed = i + 1;
            if (completed == 1 || completed % 10 == 0) {
                runOnUiThread(() -> status.setText(name + ": " + completed + "/" + RUNS));
            }
        }
        return stats;
    }

    private void parseNativeTimings(String report, Stats stats) {
        if (report == null) return;
        String[] lines = report.split("\\r?\\n");
        for (String line : lines) {
            Matcher matcher = TIME_PATTERN.matcher(line);
            if (!matcher.matches()) continue;
            String key = matcher.group(1).trim();
            double value = Double.parseDouble(matcher.group(2));
            stats.add(key, value);
        }
    }

    private String buildReport(int modeIndex,
                               Bitmap input,
                               int srcWidth,
                               int srcHeight,
                               int outWidth,
                               int outHeight,
                               Stats v1,
                               Stats v2) {
        StringBuilder out = new StringBuilder();
        out.append("Benchmark complete\n")
                .append("Runs per version: ").append(RUNS).append('\n')
                .append("Mode: ").append(DOWNSAMPLE_LABELS[modeIndex]).append('\n')
                .append("Input: ").append(input.getWidth()).append('x').append(input.getHeight()).append('\n')
                .append("Trimmed: ").append(srcWidth).append('x').append(srcHeight).append('\n')
                .append("Output: ").append(outWidth).append('x').append(outHeight).append("\n\n");

        appendMetricComparison(out, v1, v2, "Java wall total");
        appendMetricComparison(out, v1, v2, "Vulkan command submit+wait total");
        appendMetricComparison(out, v1, v2, "RGB -> YUV444 upload");
        appendMetricComparison(out, v1, v2, "GPU upload copy to DEVICE_LOCAL");
        appendMetricComparison(out, v1, v2, "Barrier upload -> shader read");
        appendMetricComparison(out, v1, v2, "Downsample");
        appendMetricComparison(out, v1, v2, "Barrier downsample -> sobel");
        appendMetricComparison(out, v1, v2, "Sobel mask");
        appendMetricComparison(out, v1, v2, "Barrier sobel -> median_v2");
        appendMetricComparison(out, v1, v2, "Median_v2");
        appendMetricComparison(out, v1, v2, "Final copy DEVICE_LOCAL -> readback");
        appendMetricComparison(out, v1, v2, "Barrier median_v2 -> CPU read");
        appendMetricComparison(out, v1, v2, "Barrier readback -> CPU read");

        MetricSummary v1Total = v1.summary("Java wall total");
        MetricSummary v2Total = v2.summary("Java wall total");
        if (v1Total != null && v2Total != null) {
            out.append("\nWinner by Java wall median: ")
                    .append(v1Total.median <= v2Total.median ? v1.name : v2.name)
                    .append('\n');
        }
        return out.toString();
    }

    private void appendMetricComparison(StringBuilder out, Stats v1, Stats v2, String key) {
        MetricSummary a = v1.summary(key);
        MetricSummary b = v2.summary(key);
        if (a == null && b == null) return;
        out.append(key).append('\n');
        if (a != null) {
            out.append("  V1 mean=").append(ms(a.mean))
                    .append(" median=").append(ms(a.median)).append('\n');
        }
        if (b != null) {
            out.append("  V2 mean=").append(ms(b.mean))
                    .append(" median=").append(ms(b.median)).append('\n');
        }
        if (a != null && b != null && a.median > 0.0 && b.median > 0.0) {
            double ratio = b.median / a.median;
            out.append("  V2/V1 median ratio=").append(String.format(Locale.US, "%.3f", ratio))
                    .append('\n');
        }
        out.append('\n');
    }

    private String ms(double value) {
        return String.format(Locale.US, "%.3f ms", value);
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

    private static final class Stats {
        final String name;
        final Map<String, List<Double>> metrics = new LinkedHashMap<>();

        Stats(String name) {
            this.name = name;
        }

        void add(String key, double value) {
            metrics.computeIfAbsent(key, ignored -> new ArrayList<>()).add(value);
        }

        MetricSummary summary(String key) {
            List<Double> values = metrics.get(key);
            if (values == null || values.isEmpty()) return null;
            ArrayList<Double> sorted = new ArrayList<>(values);
            Collections.sort(sorted);
            double sum = 0.0;
            for (double value : sorted) sum += value;
            double mean = sum / sorted.size();
            int mid = sorted.size() / 2;
            double median = sorted.size() % 2 == 0
                    ? (sorted.get(mid - 1) + sorted.get(mid)) / 2.0
                    : sorted.get(mid);
            return new MetricSummary(mean, median);
        }
    }

    private static final class MetricSummary {
        final double mean;
        final double median;

        MetricSummary(double mean, double median) {
            this.mean = mean;
            this.median = median;
        }
    }
}
