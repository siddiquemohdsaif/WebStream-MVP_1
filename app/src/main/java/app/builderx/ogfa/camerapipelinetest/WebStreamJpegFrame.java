package app.builderx.ogfa.camerapipelinetest;

public final class WebStreamJpegFrame {
    private final String participantId;
    private final byte[] jpegData;
    private final int width;
    private final int height;
    private final int frameRateFps;
    private final long timestampMs;
    private final long sequence;
    private final boolean frontCamera;
    private final int rotationDegrees;

    WebStreamJpegFrame(
            String participantId,
            byte[] jpegData,
            int width,
            int height,
            int frameRateFps,
            long timestampMs,
            long sequence,
            boolean frontCamera,
            int rotationDegrees) {
        this.participantId = participantId;
        this.jpegData = jpegData;
        this.width = width;
        this.height = height;
        this.frameRateFps = Math.max(1, frameRateFps);
        this.timestampMs = timestampMs;
        this.sequence = sequence;
        this.frontCamera = frontCamera;
        this.rotationDegrees = normalizeRotation(rotationDegrees);
    }

    public String getParticipantId() {
        return participantId;
    }

    public byte[] getJpegData() {
        return jpegData;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public int getFrameRateFps() {
        return frameRateFps;
    }

    public long getTimestampMs() {
        return timestampMs;
    }

    public long getSequence() {
        return sequence;
    }

    public boolean isFrontCamera() {
        return frontCamera;
    }

    public int getRotationDegrees() {
        return rotationDegrees;
    }

    private static int normalizeRotation(int degrees) {
        int normalized = degrees % 360;
        if (normalized < 0) normalized += 360;
        return (normalized / 90) * 90;
    }
}
