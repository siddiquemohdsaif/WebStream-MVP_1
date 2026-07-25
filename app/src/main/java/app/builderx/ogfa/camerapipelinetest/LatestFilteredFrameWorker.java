package app.builderx.ogfa.camerapipelinetest;

import android.content.Context;

public class LatestFilteredFrameWorker {
    public interface FilteredFrameSource {
        int getMaxFps();
        SendResult onLatestFilteredJpeg(byte[] jpeg, int width, int height, long timestampMs);
    }

    public static final class SendResult {
        public final boolean sent;
        public final boolean queued;
        public final long sequence;
        public final double elapsedMs;

        public SendResult(boolean sent, double elapsedMs) {
            this(sent, false, 0L, elapsedMs);
        }

        public SendResult(boolean sent, boolean queued, long sequence, double elapsedMs) {
            this.sent = sent;
            this.queued = queued;
            this.sequence = sequence;
            this.elapsedMs = elapsedMs;
        }
    }

    private final FilteredFrameSource source;

    public LatestFilteredFrameWorker(Context context, FilteredFrameSource source) {
        this.source = source;
    }

    public void start() {
        nativeStart(source);
    }

    public void stop() {
        nativeStop();
    }

    public void shutdown() {
        nativeShutdown();
    }

    private static native void nativeStart(FilteredFrameSource source);
    private static native void nativeStop();
    private static native void nativeShutdown();
}
