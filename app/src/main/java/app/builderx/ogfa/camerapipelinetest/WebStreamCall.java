package app.builderx.ogfa.camerapipelinetest;

import android.os.Handler;
import android.util.Log;

public final class WebStreamCall {
    private static final String TAG = "XX_PIPELINE_CALL";

    interface EndListener {
        void onEnded(WebStreamCall call);
    }

    public enum State {
        IDLE,
        CONNECTING,
        CONNECTED,
        LEFT,
        FAILED
    }

    private final String callId;
    private final String userId;
    private final String displayName;
    private final String authToken;
    private final String serverUrl;
    private final Handler mainHandler;
    private final WebStreamClient.Listener listener;
    private final EndListener endListener;
    private JpegWebSocketTransport transport;
    private volatile State state = State.IDLE;
    private long sequence;

    WebStreamCall(
            String callId,
            String userId,
            String displayName,
            String authToken,
            String serverUrl,
            Handler mainHandler,
            WebStreamClient.Listener listener,
            EndListener endListener) {
        this.callId = callId;
        this.userId = userId;
        this.displayName = displayName;
        this.authToken = authToken;
        this.serverUrl = serverUrl;
        this.mainHandler = mainHandler;
        this.listener = listener;
        this.endListener = endListener;
    }

    public String getCallId() {
        return callId;
    }

    public State getState() {
        return state;
    }

    void start() {
        if (state != State.IDLE) {
            return;
        }
        state = State.CONNECTING;
        dispatchConnecting();
        transport = new JpegWebSocketTransport(
                serverUrl,
                callId,
                userId,
                displayName,
                authToken,
                new JpegWebSocketTransport.Listener() {
                    @Override
                    public void onConnected() {
                        mainHandler.post(WebStreamCall.this::connectIfActive);
                    }

                    @Override
                    public void onWaitingForPeer() {
                        mainHandler.post(WebStreamCall.this::dispatchWaitingForPeer);
                    }

                    @Override
                    public void onJpegReceived(WebStreamJpegFrame frame) {
                        mainHandler.post(() -> dispatchJpegReceived(frame));
                    }

                    @Override
                    public void onClosed() {
                        mainHandler.post(WebStreamCall.this::handleClosed);
                    }

                    @Override
                    public void onError(Throwable error) {
                        mainHandler.post(() -> fail(error));
                    }
                });
        transport.connect();
    }

    public SendOutcome sendJpeg(
            byte[] jpegData,
            int width,
            int height,
            int frameRateFps,
            int bitrateKbps,
            long timestampMs) {
        JpegWebSocketTransport activeTransport = transport;
        if (state != State.CONNECTED || activeTransport == null) {
            Log.d(TAG, "webstream_call_send_drop | t=" + System.currentTimeMillis()
                    + " state=" + state
                    + " hasTransport=" + (activeTransport != null)
                    + " jpegBytes=" + (jpegData == null ? 0 : jpegData.length));
            return new SendOutcome(false, false, sequence);
        }
        long nextSequence = ++sequence;
        boolean queued = activeTransport.sendJpeg(
                jpegData,
                width,
                height,
                frameRateFps,
                bitrateKbps,
                timestampMs,
                nextSequence);
        return new SendOutcome(true, queued, nextSequence);
    }

    public void leave() {
        if (state == State.LEFT) {
            return;
        }
        state = State.LEFT;
        if (endListener != null) {
            endListener.onEnded(this);
        }
        if (transport != null) {
            transport.close();
            transport = null;
        }
        dispatchDisconnected();
    }

    private void connectIfActive() {
        if (state != State.CONNECTING) {
            return;
        }
        state = State.CONNECTED;
        if (listener != null) {
            listener.onConnected();
        }
    }

    private void handleClosed() {
        if (state == State.LEFT) {
            return;
        }
        state = State.LEFT;
        JpegWebSocketTransport closedTransport = transport;
        transport = null;
        if (endListener != null) {
            endListener.onEnded(this);
        }
        if (closedTransport != null) {
            closedTransport.close();
        }
        dispatchDisconnected();
    }

    private void fail(Throwable error) {
        if (state == State.LEFT) {
            return;
        }
        state = State.FAILED;
        if (endListener != null) {
            endListener.onEnded(this);
        }
        if (transport != null) {
            transport.close();
            transport = null;
        }
        if (listener != null) {
            listener.onError(error);
        }
    }

    private void dispatchConnecting() {
        if (listener != null) {
            mainHandler.post(listener::onConnecting);
        }
    }

    private void dispatchWaitingForPeer() {
        if (listener != null && state == State.CONNECTING) {
            listener.onWaitingForPeer();
        }
    }

    private void dispatchJpegReceived(WebStreamJpegFrame frame) {
        if (listener != null && state == State.CONNECTED) {
            listener.onJpegReceived(frame);
        }
    }

    private void dispatchDisconnected() {
        if (listener != null) {
            listener.onDisconnected();
        }
    }

    public static final class SendOutcome {
        public final boolean sent;
        public final boolean queued;
        public final long sequence;

        SendOutcome(boolean sent, boolean queued, long sequence) {
            this.sent = sent;
            this.queued = queued;
            this.sequence = sequence;
        }
    }
}
