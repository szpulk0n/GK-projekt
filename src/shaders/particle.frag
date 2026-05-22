#version 330 core
out vec4 FragColor;
in vec3 pColor;

void main()
{
    // Okragly punkt
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;
    float alpha = 1.0 - smoothstep(0.2, 0.5, dist);
    FragColor = vec4(pColor, alpha);
}
