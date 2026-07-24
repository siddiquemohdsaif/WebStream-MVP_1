package app.builderx.ogfa.camerapipelinetest;

import android.content.Intent;
import android.os.Bundle;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        findViewById(R.id.openCamera).setOnClickListener(view ->
                startActivity(new Intent(this, CameraActivity.class)));
        findViewById(R.id.openPreprocessing).setOnClickListener(view ->
                startActivity(new Intent(this, PreprocessingTest.class)));
        findViewById(R.id.openOptimizedPreprocessing).setOnClickListener(view ->
                startActivity(new Intent(this, OptimizedPreprocessingTest.class)));
        findViewById(R.id.openOptimizedPreprocessing2).setOnClickListener(view ->
                startActivity(new Intent(this, OptimizedPreprocessingTest_2.class)));
        findViewById(R.id.openPreprocessingBenchmark).setOnClickListener(view ->
                startActivity(new Intent(this, PreprocessingBenchmarkTest.class)));
        findViewById(R.id.openVulkanFeatureTest).setOnClickListener(view ->
                startActivity(new Intent(this, VulkanFeatureTestActivity.class)));
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }
}

