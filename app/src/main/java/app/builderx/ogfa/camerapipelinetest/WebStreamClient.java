package app.builderx.ogfa.camerapipelinetest;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.text.TextUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class WebStreamClient {
    private static final String DEFAULT_SERVER_URL = "ws://168.144.23.108:8080";

    private final Context applicationContext;
    private final String userId;
    private final String displayName;
    private final String authToken;
    private final String serverUrl;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final List<WebStreamCall> activeCalls = Collections.synchronizedList(new ArrayList<>());
    private boolean released;

    private WebStreamClient(Builder builder) {
        this.applicationContext = builder.context.getApplicationContext();
        this.userId = builder.userId;
        this.displayName = builder.displayName;
        this.authToken = builder.authToken;
        this.serverUrl = TextUtils.isEmpty(builder.serverUrl)
                ? DEFAULT_SERVER_URL
                : builder.serverUrl;
    }

    public String getUserId() {
        return userId;
    }

    public String getDisplayName() {
        return displayName;
    }

    public WebStreamCall joinCall(String callId, Listener listener) {
        ensureActive();
        String normalizedCallId = normalize(callId);
        if (TextUtils.isEmpty(normalizedCallId)) {
            throw new IllegalArgumentException("callId is required.");
        }

        WebStreamCall call = new WebStreamCall(
                normalizedCallId,
                userId,
                displayName,
                authToken,
                serverUrl,
                mainHandler,
                listener,
                endedCall -> activeCalls.remove(endedCall));
        activeCalls.add(call);
        call.start();
        return call;
    }

    public void release() {
        if (released) {
            return;
        }
        released = true;
        List<WebStreamCall> callsToLeave;
        synchronized (activeCalls) {
            callsToLeave = new ArrayList<>(activeCalls);
            activeCalls.clear();
        }
        for (WebStreamCall call : callsToLeave) {
            call.leave();
        }
    }

    private void ensureActive() {
        if (released) {
            throw new IllegalStateException("WebStreamClient has already been released.");
        }
    }

    private static String normalize(String value) {
        return value == null ? null : value.trim();
    }

    public interface Listener {
        void onConnecting();

        default void onWaitingForPeer() {
        }

        void onConnected();

        void onJpegReceived(WebStreamJpegFrame frame);

        void onDisconnected();

        void onError(Throwable error);
    }

    public static final class Builder {
        private final Context context;
        private String userId;
        private String displayName;
        private String authToken;
        private String serverUrl;

        public Builder(Context context) {
            if (context == null) {
                throw new IllegalArgumentException("context is required.");
            }
            this.context = context;
        }

        public Builder userId(String userId) {
            this.userId = normalize(userId);
            return this;
        }

        public Builder displayName(String displayName) {
            this.displayName = normalize(displayName);
            return this;
        }

        public Builder authToken(String authToken) {
            this.authToken = normalize(authToken);
            return this;
        }

        public Builder serverUrl(String serverUrl) {
            this.serverUrl = normalize(serverUrl);
            return this;
        }

        public WebStreamClient build() {
            if (TextUtils.isEmpty(userId)) {
                throw new IllegalArgumentException("userId is required.");
            }
            return new WebStreamClient(this);
        }
    }
}
