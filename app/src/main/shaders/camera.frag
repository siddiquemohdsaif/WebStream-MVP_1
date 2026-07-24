#version 450
layout(binding=0) uniform sampler2D yTex;
layout(binding=1) uniform sampler2D uTex;
layout(binding=2) uniform sampler2D vTex;
layout(push_constant) uniform DisplayFit {
    vec2 scale;
    int rotation;
    int mirror;
} fit;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
void main() {
    vec2 displayUv = (uv - vec2(0.5)) / fit.scale + vec2(0.5);
    if (any(lessThan(displayUv, vec2(0.0))) || any(greaterThan(displayUv, vec2(1.0)))) {
        color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Presentation order is rotate clockwise, then optionally mirror horizontally.
    if (fit.mirror != 0) displayUv.x = 1.0 - displayUv.x;

    vec2 imageUv;
    if (fit.rotation == 90) {
        imageUv = vec2(displayUv.y, 1.0 - displayUv.x);
    } else if (fit.rotation == 180) {
        imageUv = vec2(1.0 - displayUv.x, 1.0 - displayUv.y);
    } else if (fit.rotation == 270) {
        imageUv = vec2(1.0 - displayUv.y, displayUv.x);
    } else {
        imageUv = displayUv;
    }
    float y = texture(yTex, imageUv).r;
    float u = texture(uTex, imageUv).r - 0.5;
    float v = texture(vTex, imageUv).r - 0.5;
    color = vec4(y + 1.402*v, y - 0.344136*u - 0.714136*v, y + 1.772*u, 1.0);
}
