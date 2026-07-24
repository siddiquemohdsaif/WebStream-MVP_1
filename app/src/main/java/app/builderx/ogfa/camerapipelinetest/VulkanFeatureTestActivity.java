package app.builderx.ogfa.camerapipelinetest;

import android.os.Bundle;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

public class VulkanFeatureTestActivity extends AppCompatActivity implements SurfaceHolder.Callback {
    static { System.loadLibrary("camera_pipeline"); }

    private TextView status;

    private static native String nativeCheckVulkanFeatures(Surface surface);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_vulkan_feature_test);
        status = findViewById(R.id.vulkanFeatureStatus);
        SurfaceView surfaceView = findViewById(R.id.vulkanFeatureSurface);
        surfaceView.getHolder().addCallback(this);
    }

    @Override
    public void surfaceCreated(@NonNull SurfaceHolder holder) {
        runCheck(holder.getSurface());
    }

    @Override
    public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int width, int height) {
        runCheck(holder.getSurface());
    }

    @Override
    public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
        status.setText("Surface destroyed");
    }

    private void runCheck(Surface surface) {
        String result = nativeCheckVulkanFeatures(surface);
        status.setText(result);
    }
}
