#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform vec3 lightDir;
uniform vec3 viewPos;
uniform float impactProgress;   // 0.0 = brak, 1.0 = pelne uderzenie
uniform vec3 impactPoint;       // punkt uderzenia na powierzchni
uniform float craterRadius;     // promien kratera

// Prosty hash do generowania szumu
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i + vec2(0,0)), hash(i + vec2(1,0)), u.x),
        mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), u.x),
        u.y
    );
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += a * noise(p);
        p = p * 2.1 + vec2(1.7, 9.2);
        a *= 0.5;
    }
    return v;
}

// Mapa kontynentow oparta na wspolrzednych UV
float landMask(vec2 uv) {
    float u = uv.x;
    float v = uv.y;
    
    float base = fbm(uv * 3.0 + vec2(0.5, 0.3));
    float detail = fbm(uv * 8.0 + vec2(2.1, 1.7)) * 0.3;
    float combined = base + detail;
    
    // Ksztalt kontynentow - kilka "wysp" z szumem
    float continent1 = smoothstep(0.52, 0.58, combined + 
        0.15 * sin(u * 6.28 * 1.5 + 0.5) * cos(v * 6.28 * 2.0));
    float continent2 = smoothstep(0.54, 0.60, combined + 
        0.12 * sin(u * 6.28 * 2.5 + 2.0) * cos(v * 6.28 * 1.5 + 1.0));
    
    return clamp(continent1 + continent2 * 0.7, 0.0, 1.0);
}

void main()
{
    vec2 uv = TexCoord;
    vec3 norm = normalize(Normal);
    vec3 light = normalize(lightDir);
    
    // Mapa ladow i oceanow
    float land = landMask(uv);
    
    // Kolory oceanu - glebszy przy rownikach
    vec3 deepOcean = vec3(0.03, 0.12, 0.38);
    vec3 shallowOcean = vec3(0.08, 0.25, 0.55);
    float oceanDepth = fbm(uv * 5.0) * 0.5 + 0.5;
    vec3 oceanColor = mix(deepOcean, shallowOcean, oceanDepth);
    
    // Kolory ladownic - zielen, pustynie, gory
    vec3 forest = vec3(0.08, 0.28, 0.06);
    vec3 desert = vec3(0.65, 0.52, 0.22);
    vec3 mountain = vec3(0.45, 0.40, 0.35);
    float landType = fbm(uv * 6.0 + vec2(5.0));
    float mountainMask = fbm(uv * 10.0 + vec2(3.0));
    vec3 landColor = mix(forest, desert, smoothstep(0.4, 0.7, landType));
    landColor = mix(landColor, mountain, smoothstep(0.65, 0.80, mountainMask));
    
    // Czapy polarne
    float polarN = smoothstep(0.82, 0.95, uv.y);
    float polarS = smoothstep(0.18, 0.05, uv.y);
    vec3 iceColor = vec3(0.92, 0.95, 1.0);
    landColor = mix(landColor, iceColor, polarN + polarS);
    oceanColor = mix(oceanColor, iceColor * 0.9, (polarN + polarS) * 0.8);
    
    // Chmury
    float clouds = fbm(uv * 4.0 + vec2(1.3, 0.8));
    float cloudMask = smoothstep(0.52, 0.62, clouds);
    vec3 cloudColor = vec3(0.95, 0.97, 1.0);
    
    // Polacz ląd i ocean
    vec3 surfaceColor = mix(oceanColor, landColor, land);
    surfaceColor = mix(surfaceColor, cloudColor, cloudMask * 0.75);
    
    // Oswietlenie Phonga
    float diff = max(dot(norm, light), 0.0);
    float ambient = 0.08;
    
    // Specular - tylko na oceanach
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-light, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    float specular = spec * (1.0 - land) * (1.0 - cloudMask) * 0.6;
    
    vec3 finalColor = surfaceColor * (ambient + diff) + vec3(specular);
    
    // Nocna strona - swiatla miast
    if (diff < 0.05) {
        float cityLights = fbm(uv * 12.0 + vec2(7.3));
        float cityMask = land * smoothstep(0.62, 0.72, cityLights) * (1.0 - cloudMask);
        finalColor += cityMask * vec3(1.0, 0.85, 0.4) * (1.0 - diff / 0.05) * 0.3;
    }
    
    // === EFEKT UDERZENIA ===
    if (impactProgress > 0.0) {
        // Odleglosc od punktu uderzenia na sferze
        vec3 fragNorm = normalize(FragPos);
        vec3 impNorm = normalize(impactPoint);
        float angleDist = acos(clamp(dot(fragNorm, impNorm), -1.0, 1.0));
        float distFromImpact = angleDist; // w radianach
        
        float craterRad = craterRadius;
        
        // Krater - ciemna dziura
        float craterMask = smoothstep(craterRad * 0.8, craterRad * 0.3, distFromImpact) * impactProgress;
        vec3 craterColor = vec3(0.15, 0.05, 0.02);
        finalColor = mix(finalColor, craterColor, craterMask);
        
        // Rozgrzany brzeg kratera
        float rimMask = smoothstep(craterRad * 1.5, craterRad * 0.85, distFromImpact) *
                        smoothstep(craterRad * 0.3, craterRad * 0.7, distFromImpact) * impactProgress;
        finalColor += rimMask * vec3(1.0, 0.4, 0.05) * 2.0;
        
        // Fala uderzeniowa - rozchodzi sie
        float shockwaveRadius = impactProgress * 1.8;
        float shockwave = smoothstep(shockwaveRadius + 0.08, shockwaveRadius, distFromImpact) *
                          smoothstep(shockwaveRadius - 0.25, shockwaveRadius, distFromImpact) *
                          (1.0 - impactProgress * 0.7);
        finalColor += shockwave * vec3(1.0, 0.7, 0.3) * 1.5;
        
        // Globalne ogrzanie planety
        finalColor = mix(finalColor, finalColor + vec3(0.3, 0.05, 0.0), impactProgress * 0.4);
    }
    
    // Atmosfera - niebieska poswata na krawedziach
    float rimFactor = 1.0 - max(dot(norm, viewDir), 0.0);
    rimFactor = pow(rimFactor, 3.0);
    vec3 atmoColor = vec3(0.2, 0.5, 1.0) * rimFactor * 0.7;
    finalColor += atmoColor;
    
    FragColor = vec4(finalColor, 1.0);
}
