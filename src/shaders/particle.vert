#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aSize;

out vec3 pColor;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    pColor = aColor;
    gl_Position = projection * view * vec4(aPos, 1.0);
    gl_PointSize = max(1.0, aSize / gl_Position.w * 100.0);
}
