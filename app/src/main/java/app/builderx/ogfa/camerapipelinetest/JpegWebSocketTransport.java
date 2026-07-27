package app.builderx.ogfa.camerapipelinetest;

import android.text.TextUtils;
import android.util.Base64;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
import okio.ByteString;

final class JpegWebSocketTransport {
    interface Listener {
        void onConnected();

        void onWaitingForPeer();

        void onJpegReceived(WebStreamJpegFrame frame);

        void onClosed();

        void onError(Throwable error);
    }

    private static final String TAG = "XX_PIPELINE_CALL";
    private static final int NORMAL_CLOSE = 1000;
    private static final int VIDEO_PACKET_TYPE = 1;
    private static final int VIDEO_PACKET_HEADER_BYTES = 33;
    private static final int JPEG_FORMAT_VALUE = 1;
    private static final int CAMERA_FRONT_FLAG = 1 << 30;
    private static final int CAMERA_ROTATION_SHIFT = 28;
    private static final int CAMERA_ROTATION_MASK = 3 << CAMERA_ROTATION_SHIFT;
    private static final int BITRATE_METADATA_MASK = (1 << CAMERA_ROTATION_SHIFT) - 1;

    private final OkHttpClient okHttpClient = new OkHttpClient.Builder().build();
    private final String serverUrl;
    private final String callId;
    private final String userId;
    private final String displayName;
    private final String authToken;
    private final Listener listener;
    private final Map<Integer, String> participantIdsByCUuid = new HashMap<>();

    private volatile WebSocket webSocket;
    private volatile boolean closed;
    private volatile boolean joined;
    private volatile boolean peerConnected;
    private volatile int cUuid;

    JpegWebSocketTransport(
            String serverUrl,
            String callId,
            String userId,
            String displayName,
            String authToken,
            Listener listener) {
        this.serverUrl = serverUrl;
        this.callId = callId;
        this.userId = userId;
        this.displayName = displayName;
        this.authToken = authToken;
        this.listener = listener;
    }

    void connect() {
        Request request = new Request.Builder()
                .url(serverUrl)
                .build();
        webSocket = okHttpClient.newWebSocket(request, new WebSocketListener() {
            @Override
            public void onOpen(WebSocket webSocket, Response response) {
                Log.d(TAG, "websocket_opened | t=" + System.currentTimeMillis());
                sendJoin();
            }

            @Override
            public void onMessage(WebSocket webSocket, String text) {
                handleTextMessage(text);
            }

            @Override
            public void onMessage(WebSocket webSocket, ByteString bytes) {
                handleBinaryMessage(bytes);
            }

            @Override
            public void onClosed(WebSocket webSocket, int code, String reason) {
                Log.d(TAG, "websocket_closed | t=" + System.currentTimeMillis()
                        + " code=" + code
                        + " reason=" + reason);
                if (!closed && listener != null) {
                    listener.onClosed();
                }
            }

            @Override
            public void onFailure(WebSocket webSocket, Throwable t, Response response) {
                Log.d(TAG, "websocket_failure | t=" + System.currentTimeMillis(), t);
                if (!closed && listener != null) {
                    listener.onError(t);
                }
            }
        });
    }

    boolean sendJpeg(
            byte[] jpegData,
            int width,
            int height,
            int frameRateFps,
            int bitrateKbps,
            long timestampMs,
            long sequence,
            boolean frontCamera,
            int rotationDegrees) {
        if (webSocket == null || closed || !joined || !peerConnected || cUuid == 0
                || jpegData == null || jpegData.length == 0) {
            Log.d(TAG, "send_jpeg_drop | t=" + System.currentTimeMillis()
                    + " socket=" + (webSocket != null)
                    + " closed=" + closed
                    + " joined=" + joined
                    + " peerConnected=" + peerConnected
                    + " cUuid=" + cUuid
                    + " jpegBytes=" + (jpegData == null ? 0 : jpegData.length));
            return false;
        }

        ByteBuffer packet = ByteBuffer
                .allocate(VIDEO_PACKET_HEADER_BYTES + jpegData.length)
                .order(ByteOrder.BIG_ENDIAN);
        packet.putShort((short) VIDEO_PACKET_TYPE);
        packet.putInt(cUuid);
        packet.putLong(timestampMs);
        packet.put((byte) JPEG_FORMAT_VALUE);
        packet.putShort((short) clampUnsignedShort(width));
        packet.putShort((short) clampUnsignedShort(height));
        packet.putShort((short) clampUnsignedShort(frameRateFps));
        int metadataFlags = Math.max(0, bitrateKbps) & BITRATE_METADATA_MASK;
        if (frontCamera) metadataFlags |= CAMERA_FRONT_FLAG;
        metadataFlags |= rotationCode(rotationDegrees) << CAMERA_ROTATION_SHIFT;
        packet.putInt(metadataFlags);
        packet.putInt((int) sequence);
        packet.putInt(jpegData.length);
        packet.put(jpegData);
        return webSocket.send(ByteString.of(packet.array()));
    }

    void close() {
        if (webSocket != null) {
            sendLeave();
            closed = true;
            if (!webSocket.close(NORMAL_CLOSE, "jpeg transport closed")) {
                webSocket.cancel();
            }
            webSocket = null;
        } else {
            closed = true;
        }
        okHttpClient.dispatcher().executorService().shutdown();
        okHttpClient.connectionPool().evictAll();
    }

    private void sendJoin() {
        try {
            JSONObject message = new JSONObject()
                    .put("type", "client.join")
                    .put("callId", callId)
                    .put("userId", userId)
                    .put("mediaCapabilities", new JSONObject()
                            .put("preferredImageFormat", "jpeg")
                            .put("imageFormats", new JSONArray().put("jpeg")));
            if (!TextUtils.isEmpty(displayName)) {
                message.put("displayName", displayName);
            }
            if (!TextUtils.isEmpty(authToken)) {
                message.put("authToken", authToken);
            }
            sendJson(message);
        } catch (JSONException error) {
            reportError(error);
        }
    }

    private void sendLeave() {
        try {
            sendJson(new JSONObject()
                    .put("type", "client.leave")
                    .put("callId", callId)
                    .put("userId", userId));
        } catch (JSONException error) {
                Log.d(TAG, "leave_message_error | t=" + System.currentTimeMillis()
                        + " message=" + error.getMessage());
        }
    }

    private void sendJson(JSONObject message) {
        if (webSocket != null && !closed) {
            webSocket.send(message.toString());
        }
    }

    private void handleTextMessage(String text) {
        try {
            JSONObject message = new JSONObject(text);
            String type = message.optString("type");
            if ("server.joined".equals(type)) {
                joined = true;
                cUuid = message.optInt("cUuid", 0);
                Log.d(TAG, "server_joined | t=" + System.currentTimeMillis()
                        + " cUuid=" + cUuid
                        + " message=" + text);
                if (cUuid != 0) {
                    participantIdsByCUuid.put(cUuid, userId);
                }
                rememberParticipantMappings(message.optJSONArray("participants"));
                updatePeerConnectionFromCurrentParticipants();
            } else if ("server.participant_joined".equals(type)) {
                JSONObject participant = message.optJSONObject("participant");
                rememberParticipantMapping(participant);
                if (isRemoteParticipant(participant)) {
                    peerConnected = true;
                    Log.d(TAG, "remote_peer_connected | t=" + System.currentTimeMillis()
                            + " participant=" + participant);
                    if (listener != null) {
                        listener.onConnected();
                    }
                }
            } else if ("server.participant_left".equals(type)) {
                JSONObject participant = message.optJSONObject("participant");
                boolean remoteLeft = isRemoteParticipant(participant);
                removeParticipantMapping(participant);
                if (remoteLeft && listener != null) {
                    listener.onClosed();
                }
            } else if ("server.media.video".equals(type)) {
                handleJsonJpegFrame(message);
            } else if ("server.left".equals(type) && listener != null) {
                listener.onClosed();
            } else if ("server.error".equals(type)) {
                reportError(new IllegalStateException(
                        message.optString("message", "Server rejected JPEG transport request.")));
            }
        } catch (JSONException error) {
            reportError(error);
        }
    }

    private void handleBinaryMessage(ByteString bytes) {
        if (bytes == null || bytes.size() < VIDEO_PACKET_HEADER_BYTES) {
            return;
        }
        ByteBuffer packet = ByteBuffer.wrap(bytes.toByteArray()).order(ByteOrder.BIG_ENDIAN);
        while (packet.remaining() >= VIDEO_PACKET_HEADER_BYTES) {
            int packetType = Short.toUnsignedInt(packet.getShort());
            int frameCUuid = packet.getInt();
            long timestampMs = packet.getLong();
            int format = Byte.toUnsignedInt(packet.get());
            int width = Short.toUnsignedInt(packet.getShort());
            int height = Short.toUnsignedInt(packet.getShort());
            int frameRateFps = Short.toUnsignedInt(packet.getShort());
            int metadataFlags = packet.getInt();
            long sequence = Integer.toUnsignedLong(packet.getInt());
            int payloadLength = packet.getInt();

            if (packetType != VIDEO_PACKET_TYPE
                    || format != JPEG_FORMAT_VALUE
                    || payloadLength <= 0
                    || payloadLength > packet.remaining()) {
                return;
            }

            byte[] jpegData = new byte[payloadLength];
            packet.get(jpegData);
            dispatchJpegFrame(frameCUuid, jpegData, width, height, frameRateFps,
                    timestampMs, sequence, (metadataFlags & CAMERA_FRONT_FLAG) != 0,
                    rotationFromMetadata(metadataFlags));
        }
    }

    private void handleJsonJpegFrame(JSONObject message) {
        String participantId = message.optString("userId");
        if (TextUtils.isEmpty(participantId) || userId.equals(participantId)) {
            return;
        }
        if (!"jpeg".equalsIgnoreCase(message.optString("format", "jpeg"))) {
            return;
        }
        String data = message.optString("data");
        if (TextUtils.isEmpty(data)) {
            return;
        }
        try {
            byte[] jpegData = Base64.decode(data, Base64.DEFAULT);
            if (listener != null) {
                listener.onJpegReceived(new WebStreamJpegFrame(
                        participantId,
                        jpegData,
                        message.optInt("width", 0),
                        message.optInt("height", 0),
                        message.optInt("frameRateFps", 15),
                        message.optLong("timestampMs", 0L),
                        message.optLong("sequence", 0L),
                        message.optBoolean("frontCamera", false),
                        message.optInt("rotationDegrees", 0)));
            }
        } catch (IllegalArgumentException error) {
            reportError(error);
        }
    }

    private void dispatchJpegFrame(
            int frameCUuid,
            byte[] jpegData,
            int width,
            int height,
            int frameRateFps,
            long timestampMs,
            long sequence,
            boolean frontCamera,
            int rotationDegrees) {
        if (frameCUuid == cUuid || listener == null) {
            return;
        }
        String participantId = participantIdsByCUuid.get(frameCUuid);
        if (TextUtils.isEmpty(participantId)) {
            return;
        }
        listener.onJpegReceived(new WebStreamJpegFrame(
                participantId,
                jpegData,
                width,
                height,
                frameRateFps,
                timestampMs,
                sequence,
                frontCamera,
                rotationDegrees));
    }

    private static int rotationCode(int rotationDegrees) {
        int normalized = rotationDegrees % 360;
        if (normalized < 0) normalized += 360;
        return (normalized / 90) & 0x3;
    }

    private static int rotationFromMetadata(int metadataFlags) {
        return ((metadataFlags & CAMERA_ROTATION_MASK) >> CAMERA_ROTATION_SHIFT) * 90;
    }

    private void rememberParticipantMappings(JSONArray participants) {
        if (participants == null) {
            return;
        }
        for (int i = 0; i < participants.length(); i++) {
            rememberParticipantMapping(participants.optJSONObject(i));
        }
    }

    private void updatePeerConnectionFromCurrentParticipants() {
        peerConnected = false;
        for (Integer participantCUuid : participantIdsByCUuid.keySet()) {
            if (participantCUuid != null && participantCUuid != 0 && participantCUuid != cUuid) {
                peerConnected = true;
                break;
            }
        }
        if (listener == null) {
            return;
        }
        if (peerConnected) {
            listener.onConnected();
        } else {
            listener.onWaitingForPeer();
        }
    }

    private void rememberParticipantMapping(JSONObject participant) {
        if (participant == null) {
            return;
        }
        String participantId = participant.optString("userId", null);
        int participantCUuid = participant.optInt("cUuid", 0);
        if (!TextUtils.isEmpty(participantId) && participantCUuid != 0) {
            participantIdsByCUuid.put(participantCUuid, participantId);
        }
    }

    private boolean isRemoteParticipant(JSONObject participant) {
        if (participant == null) {
            return false;
        }
        int participantCUuid = participant.optInt("cUuid", 0);
        if (participantCUuid != 0) {
            return participantCUuid != cUuid;
        }
        String participantId = participant.optString("userId", null);
        return !TextUtils.isEmpty(participantId) && !userId.equals(participantId);
    }

    private void removeParticipantMapping(JSONObject participant) {
        if (participant == null) {
            return;
        }
        int participantCUuid = participant.optInt("cUuid", 0);
        if (participantCUuid != 0) {
            participantIdsByCUuid.remove(participantCUuid);
            return;
        }
        String participantId = participant.optString("userId", null);
        Integer cUuidToRemove = null;
        for (Map.Entry<Integer, String> entry : participantIdsByCUuid.entrySet()) {
            if (entry.getValue().equals(participantId)) {
                cUuidToRemove = entry.getKey();
                break;
            }
        }
        if (cUuidToRemove != null) {
            participantIdsByCUuid.remove(cUuidToRemove);
        }
    }

    private int clampUnsignedShort(int value) {
        if (value < 0) {
            return 0;
        }
        return Math.min(value, 0xffff);
    }

    private void reportError(Throwable error) {
        if (listener != null) {
            listener.onError(error);
        }
    }
}
