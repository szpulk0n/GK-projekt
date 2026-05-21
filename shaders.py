# shaders.py
# Kody shaderów GLSL dla cząsteczek, planet, linii wektorów i tła.

# Prosty Vertex Shader dla planet
BASIC_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 ourColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    ourColor = aColor;
}
"""

# Prosty Fragment Shader dla planet
BASIC_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
in vec3 ourColor;

void main()
{
    FragColor = vec4(ourColor, 1.0);
}
"""

# Vertex Shader dla cząsteczek pyłu (Particles)
PARTICLE_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aSpeed; // Prędkość do kolorowania

out float vSpeed;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * vec4(aPos, 1.0);
    // Skalowanie wielkości punktu w zależności od odległości opcjonalnie, tu ustawiamy stały rozmiar bazy
    gl_PointSize = 8.0 / gl_Position.z; 
    vSpeed = aSpeed;
}
"""

# Fragment Shader dla cząsteczek pyłu wykorzystujący gl_PointCoord
PARTICLE_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
in float vSpeed;

void main()
{
    // Miękkie koło z użyciem gl_PointCoord (0.0 do 1.0)
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if(dist > 0.5)
        discard;
        
    // Miękkie krawędzie
    float alpha = smoothstep(0.5, 0.1, dist);
    
    // Zmiana koloru w zależności od prędkości
    vec3 color = mix(vec3(0.2, 0.4, 1.0), vec3(1.0, 0.4, 0.2), clamp(vSpeed * 0.1, 0.0, 1.0));
    
    FragColor = vec4(color, alpha * 0.8);
}
"""
