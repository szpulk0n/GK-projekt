#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightDir;
uniform vec3 viewPos;
uniform float cometGlow;

float hashf(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123);
}

float noiseRock(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float v000 = hashf(i);
    float v100 = hashf(i + vec3(1,0,0));
    float v010 = hashf(i + vec3(0,1,0));
    float v110 = hashf(i + vec3(1,1,0));
    float v001 = hashf(i + vec3(0,0,1));
    float v101 = hashf(i + vec3(1,0,1));
    float v011 = hashf(i + vec3(0,1,1));
    float v111 = hashf(i + vec3(1,1,1));
    return mix(
        mix(mix(v000, v100, u.x), mix(v010, v110, u.x), u.y),
        mix(mix(v001, v101, u.x), mix(v011, v111, u.x), u.y),
        u.z
    );
}

void main()
{
    vec3 norm = normalize(Normal);
    vec3 light = normalize(lightDir);
    vec3 viewDir = normalize(viewPos - FragPos);

    float n1 = noiseRock(FragPos * 2.0);
    float n2 = noiseRock(FragPos * 5.0) * 0.5;
    float n3 = noiseRock(FragPos * 11.0) * 0.25;
    float rockNoise = n1 + n2 + n3;

    vec3 darkRock  = vec3(0.18, 0.14, 0.12);
    vec3 lightRock = vec3(0.45, 0.38, 0.30);
    vec3 iceVein   = vec3(0.65, 0.75, 0.85);

    float rockBlend = smoothstep(0.35, 0.75, rockNoise);
    float iceBlend  = smoothstep(0.78, 0.90, rockNoise);

    vec3 baseColor = mix(darkRock, lightRock, rockBlend);
    baseColor = mix(baseColor, iceVein, iceBlend);

    float diff = max(dot(norm, light), 0.0);
    float ambient = 0.12;

    vec3 reflectDir = reflect(-light, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16.0);
    float specular = spec * iceBlend * 0.4;

    vec3 finalColor = baseColor * (ambient + diff * 0.9) + vec3(specular);

    float rimFactor = 1.0 - max(dot(norm, viewDir), 0.0);
    rimFactor = pow(rimFactor, 2.0);
    vec3 glowColor = mix(vec3(0.8, 0.5, 0.1), vec3(1.0, 0.2, 0.0), cometGlow);
    finalColor += rimFactor * glowColor * (0.3 + cometGlow * 1.2);

    FragColor = vec4(finalColor, 1.0);
}
