#include "gs_glrenderer.h"

#include <glad/gl.h>

#include <cstdio>

// Fullscreen triangle: no vertex buffer, the vertex id picks the corner.
static const char * kBlitVertexSource = R"(#version 460 core
out vec2 uv;
void main() {
    uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
)";

static const char * kBlitFragmentSource = R"(#version 460 core
in vec2 uv;
out vec4 colour;
uniform sampler2D image;
void main() {
    colour = texture(image, uv);
}
)";

/*
===================
===================
*/
static GLuint compile_shader( GLenum type, const char * source ) {
    GLuint shader = glCreateShader( type );
    glShaderSource( shader, 1, &source, nullptr );
    glCompileShader( shader );

    GLint ok = GL_FALSE;
    glGetShaderiv( shader, GL_COMPILE_STATUS, &ok );
    if ( !ok ) {
        char log[1024];
        glGetShaderInfoLog( shader, sizeof( log ), nullptr, log );
        printf( "shader compile failed: %s\n", log );
        glDeleteShader( shader );
        return 0;
    }
    return shader;
}

/*
===================
===================
*/
static GLuint build_blit_program() {
    GLuint vertex = compile_shader( GL_VERTEX_SHADER, kBlitVertexSource );
    GLuint fragment = compile_shader( GL_FRAGMENT_SHADER, kBlitFragmentSource );
    if ( !vertex || !fragment ) {
        glDeleteShader( vertex );
        glDeleteShader( fragment );
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader( program, vertex );
    glAttachShader( program, fragment );
    glLinkProgram( program );
    glDeleteShader( vertex );
    glDeleteShader( fragment );

    GLint ok = GL_FALSE;
    glGetProgramiv( program, GL_LINK_STATUS, &ok );
    if ( !ok ) {
        char log[1024];
        glGetProgramInfoLog( program, sizeof( log ), nullptr, log );
        printf( "program link failed: %s\n", log );
        glDeleteProgram( program );
        return 0;
    }
    return program;
}

/*
===================
===================
*/
bool gs_glrenderer_init( GsGlRenderer * renderer ) {
    renderer->program = 0;
    renderer->vao = 0;
    renderer->target = 0;
    renderer->target_width = 0;
    renderer->target_height = 0;

    renderer->program = build_blit_program();
    if ( !renderer->program ) {
        return false;
    }

    // Core profile still requires a bound VAO, even for an attribute-less draw.
    GLuint vao = 0;
    glGenVertexArrays( 1, &vao );
    renderer->vao = vao;
    return true;
}

/*
===================
===================
*/
void gs_glrenderer_shutdown( GsGlRenderer * renderer ) {
    if ( renderer->target ) {
        glDeleteTextures( 1, &renderer->target );
        renderer->target = 0;
    }
    if ( renderer->vao ) {
        glDeleteVertexArrays( 1, &renderer->vao );
        renderer->vao = 0;
    }
    if ( renderer->program ) {
        glDeleteProgram( renderer->program );
        renderer->program = 0;
    }
    renderer->target_width = 0;
    renderer->target_height = 0;
}

/*
===================
===================
*/
bool gs_glrenderer_resize_target( GsGlRenderer * renderer, int width, int height ) {
    if ( width <= 0 || height <= 0 ) {
        return false;
    }
    if ( renderer->target && width == renderer->target_width && height == renderer->target_height ) {
        return false;
    }

    if ( renderer->target ) {
        glDeleteTextures( 1, &renderer->target );
        renderer->target = 0;
    }

    GLuint texture = 0;
    glGenTextures( 1, &texture );
    glBindTexture( GL_TEXTURE_2D, texture );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr );
    glBindTexture( GL_TEXTURE_2D, 0 );

    renderer->target = texture;
    renderer->target_width = width;
    renderer->target_height = height;
    return true;
}

/*
===================
===================
*/
void gs_glrenderer_draw( const GsGlRenderer * renderer, int viewport_width, int viewport_height ) {
    glViewport( 0, 0, viewport_width, viewport_height );
    glClearColor( 0.05f, 0.05f, 0.07f, 1.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    if ( !renderer->target ) {
        return;
    }

    glUseProgram( renderer->program );
    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, renderer->target );
    glUniform1i( glGetUniformLocation( renderer->program, "image" ), 0 );
    glBindVertexArray( renderer->vao );
    glDrawArrays( GL_TRIANGLES, 0, 3 );
    glBindVertexArray( 0 );
}
