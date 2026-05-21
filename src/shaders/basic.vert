#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aOffset;

out vec3 vertexColor;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    // Scale the base cube down to 20% size and translate by instance offset
    vec3 worldPos = (aPos * 0.2) + aOffset;
    gl_Position = projection * view * vec4(worldPos, 1.0);
    
    // Distance from center
    float dist = length(aOffset);
    
    // Normalize distance from roughly 5 to 50
    float t = clamp((dist - 5.0) / 45.0, 0.0, 1.0);
    
    // Color mapping: Inner = Yellow, Middle = Orange, Outer = Deep Blue
    vec3 colorClose = vec3(1.0, 0.9, 0.2);
    vec3 colorMid = vec3(1.0, 0.3, 0.0);
    vec3 colorFar = vec3(0.1, 0.1, 0.8);
    
    vec3 finalColor;
    if (t < 0.5) {
        finalColor = mix(colorClose, colorMid, t * 2.0);
    } else {
        finalColor = mix(colorMid, colorFar, (t - 0.5) * 2.0);
    }
    
    vertexColor = finalColor;
}
