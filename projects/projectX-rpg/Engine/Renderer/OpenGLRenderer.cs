using System.Numerics;
using System.Runtime.InteropServices;
using Engine.Renderer;
using Silk.NET.OpenGL;

namespace Engine {
    [StructLayout( LayoutKind.Sequential )]
    public readonly record struct Color( byte R, byte G, byte B, byte A = 255 ) {
        public static readonly Color White = new( 255, 255, 255 );
        public static readonly Color Black = new( 0, 0, 0 );
        public static readonly Color Red = new( 230, 41, 55 );
        public static readonly Color Green = new( 0, 228, 48 );
        public static readonly Color Blue = new( 0, 121, 241 );
        public static readonly Color Yellow = new( 253, 249, 0 );
        public static readonly Color Gray = new( 130, 130, 130 );
        public static readonly Color DarkGray = new( 40, 40, 40 );
        public static readonly Color Transparent = new( 0, 0, 0, 0 );

        public Color WithAlpha( byte a ) => this with { A = a };
    }

    public readonly record struct Rect( float X, float Y, float W, float H ) {
        public Vector2 Position => new( X, Y );
        public Vector2 Size => new( W, H );
    }

    // Position = world point at the centre of the screen. Zoom = screen pixels per world unit.
    public struct Camera2D {
        public Vector2  Position;
        public float    Zoom;

        public Camera2D( Vector2 position, float zoom = 1 ) { Position = position; Zoom = zoom; }

        public Matrix4x4 ViewMatrix( Vector2 screenSize ) =>
            Matrix4x4.CreateTranslation( -Position.X, -Position.Y, 0 ) *
            Matrix4x4.CreateScale( Zoom, Zoom, 1 ) *
            Matrix4x4.CreateTranslation( screenSize.X / 2, screenSize.Y / 2, 0 );

        public Vector2 ScreenToWorld( Vector2 screen, Vector2 screenSize ) => ( screen - screenSize / 2 ) / Zoom + Position;
        public Vector2 WorldToScreen( Vector2 world, Vector2 screenSize ) => ( world - Position ) * Zoom + screenSize / 2;

        // World-space rectangle currently visible, handy for culling.
        public Rect VisibleBounds( Vector2 screenSize ) {
            Vector2 half = screenSize / 2 / Zoom;
            return new Rect( Position.X - half.X, Position.Y - half.Y, half.X * 2, half.Y * 2 );
        }
    }

    public sealed class OpenGLTexture2D : IDisposable {
        public static readonly OpenGLTexture2D InvalidTexture = new OpenGLTexture2D(0, 0, 0);

        public uint Handle { get; }
        public int  Width { get; }
        public int  Height { get; }

        internal OpenGLTexture2D( uint handle, int width, int height ) {
            Handle = handle; Width = width; Height = height;
        }

        public void Dispose() => OpenGLInstance.Get().DeleteTexture( Handle );
    }

    // Immediate-mode 2D batch renderer. All draws between BeginFrame/EndFrame are collected into one
    // vertex buffer and flushed when the texture or camera changes, or the buffer fills up.
    // Coordinates: y points down. Without a camera, units are screen pixels with (0,0) at top-left.
    public sealed class OpenGLRenderer : IDisposable {
        [StructLayout( LayoutKind.Sequential )]
        private struct Vertex {
            public Vector2  Position;
            public Vector2  UV;
            public Color    Color;
        }

        private const int MaxVertices = 6 * 16384;

        private const string VertexShaderSource = @"#version 330 core
layout (location = 0) in vec2 a_position;
layout (location = 1) in vec2 a_uv;
layout (location = 2) in vec4 a_color;
uniform mat4 u_viewProj;
out vec2 v_uv;
out vec4 v_color;
void main() {
    v_uv = a_uv;
    v_color = a_color;
    gl_Position = u_viewProj * vec4(a_position, 0.0, 1.0);
}";

        private const string FragmentShaderSource = @"#version 330 core
in vec2 v_uv;
in vec4 v_color;
uniform sampler2D u_texture;
out vec4 fragColor;
void main() {
    fragColor = texture(u_texture, v_uv) * v_color;
}";

        public int DrawCallsLastFrame { get; private set; }

        private readonly OpenGLWindow window;
        private readonly GL         gl;
        private readonly uint       vao;
        private readonly uint       vbo;
        private readonly uint       program;
        private readonly int        viewProjLocation;
        private readonly OpenGLTexture2D    whiteTexture;

        private readonly Vertex[]   vertices = new Vertex[ MaxVertices ];
        private int                 vertexCount;
        private OpenGLTexture2D             currentTexture;
        private Camera2D?           camera;
        private int                 drawCalls;

        public OpenGLRenderer( OpenGLWindow window ) {
            this.window = window;
            gl = window.GL;

            program = CreateProgram( VertexShaderSource, FragmentShaderSource );
            viewProjLocation = gl.GetUniformLocation( program, "u_viewProj" );
            gl.UseProgram( program );
            gl.Uniform1( gl.GetUniformLocation( program, "u_texture" ), 0 );

            vao = gl.GenVertexArray();
            vbo = gl.GenBuffer();
            gl.BindVertexArray( vao );
            gl.BindBuffer( BufferTargetARB.ArrayBuffer, vbo );
            unsafe {
                uint stride = (uint)sizeof( Vertex );
                gl.BufferData( BufferTargetARB.ArrayBuffer, (nuint)( MaxVertices * stride ), null, BufferUsageARB.DynamicDraw );
                gl.EnableVertexAttribArray( 0 );
                gl.VertexAttribPointer( 0, 2, VertexAttribPointerType.Float, false, stride, (void*)0 );
                gl.EnableVertexAttribArray( 1 );
                gl.VertexAttribPointer( 1, 2, VertexAttribPointerType.Float, false, stride, (void*)8 );
                gl.EnableVertexAttribArray( 2 );
                gl.VertexAttribPointer( 2, 4, VertexAttribPointerType.UnsignedByte, true, stride, (void*)16 );
            }
            gl.BindVertexArray( 0 );

            whiteTexture = CreateTexture( 1, 1, new byte[] { 255, 255, 255, 255 } );
            currentTexture = whiteTexture;

            gl.Enable( EnableCap.Blend );
            gl.BlendFunc( BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha );
            gl.Disable( EnableCap.DepthTest );
            gl.Disable( EnableCap.CullFace );
        }

        // ---------------------------------------------------------------- Frame

        public void BeginFrame( Color clearColor ) {
            gl.ClearColor( clearColor.R / 255f, clearColor.G / 255f, clearColor.B / 255f, clearColor.A / 255f );
            gl.Clear( ClearBufferMask.ColorBufferBit );
            vertexCount = 0;
            drawCalls = 0;
            camera = null;
            currentTexture = whiteTexture;
        }

        public void EndFrame() {
            Flush();
            DrawCallsLastFrame = drawCalls;
        }

        // Pass null to draw in screen space (e.g. UI).
        public void SetCamera( Camera2D? newCamera ) {
            Flush();
            camera = newCamera;
        }

        // ---------------------------------------------------------------- Textures

        // rgba: width * height * 4 bytes, rows top to bottom.
        public OpenGLTexture2D CreateTexture( int width, int height, ReadOnlySpan<byte> rgba, bool pixelated = true ) {
            if ( rgba.Length != width * height * 4 ) {
                throw new ArgumentException( $"Expected {width * height * 4} bytes of RGBA data, got {rgba.Length}" );
            }

            uint handle = gl.GenTexture();
            gl.BindTexture( TextureTarget.Texture2D, handle );
            gl.PixelStore( PixelStoreParameter.UnpackAlignment, 1 );
            gl.TexImage2D<byte>( TextureTarget.Texture2D, 0, InternalFormat.Rgba8, (uint)width, (uint)height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, rgba );

            int filter = (int)( pixelated ? TextureMinFilter.Nearest : TextureMinFilter.Linear );
            gl.TexParameter( TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, filter );
            gl.TexParameter( TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, filter );
            gl.TexParameter( TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge );
            gl.TexParameter( TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge );

            return new OpenGLTexture2D( handle, width, height );
        }

        // ---------------------------------------------------------------- Shapes

        public void DrawRect( Vector2 position, Vector2 size, Color color ) {
            DrawQuad( whiteTexture, position, size, Vector2.Zero, Vector2.One, color );
        }

        public void DrawRect( Rect rect, Color color ) => DrawRect( rect.Position, rect.Size, color );

        public void DrawRectLines( Vector2 position, Vector2 size, float thickness, Color color ) {
            DrawRect( position, new Vector2( size.X, thickness ), color );
            DrawRect( new Vector2( position.X, position.Y + size.Y - thickness ), new Vector2( size.X, thickness ), color );
            DrawRect( new Vector2( position.X, position.Y + thickness ), new Vector2( thickness, size.Y - thickness * 2 ), color );
            DrawRect( new Vector2( position.X + size.X - thickness, position.Y + thickness ), new Vector2( thickness, size.Y - thickness * 2 ), color );
        }

        public void DrawLine( Vector2 a, Vector2 b, float thickness, Color color ) {
            Vector2 dir = b - a;
            float length = dir.Length();
            if ( length <= 0 ) return;
            Vector2 n = new Vector2( -dir.Y, dir.X ) / length * ( thickness / 2 );

            SetTexture( whiteTexture );
            Reserve( 6 );
            PushTriangle( a + n, b + n, b - n, Vector2.Zero, Vector2.Zero, Vector2.Zero, color );
            PushTriangle( a + n, b - n, a - n, Vector2.Zero, Vector2.Zero, Vector2.Zero, color );
        }

        public void DrawCircle( Vector2 center, float radius, Color color, int segments = 32 ) {
            SetTexture( whiteTexture );
            Reserve( segments * 3 );
            float step = MathF.Tau / segments;
            Vector2 prev = center + new Vector2( radius, 0 );
            for ( int i = 1; i <= segments; i++ ) {
                Vector2 next = center + new Vector2( MathF.Cos( i * step ), MathF.Sin( i * step ) ) * radius;
                PushTriangle( center, prev, next, Vector2.Zero, Vector2.Zero, Vector2.Zero, color );
                prev = next;
            }
        }

        // ---------------------------------------------------------------- Sprites

        // source: pixel rectangle inside the texture, or null for the whole texture.
        public void DrawTexture( OpenGLTexture2D texture, Vector2 position, Vector2 size, Color tint, Rect? source = null ) {
            Vector2 uv0 = Vector2.Zero;
            Vector2 uv1 = Vector2.One;
            if ( source is Rect src ) {
                Vector2 texSize = new Vector2( texture.Width, texture.Height );
                uv0 = src.Position / texSize;
                uv1 = ( src.Position + src.Size ) / texSize;
            }
            DrawQuad( texture, position, size, uv0, uv1, tint );
        }

        public void DrawTexture( OpenGLTexture2D texture, Vector2 position, Vector2 size ) => DrawTexture( texture, position, size, Color.White );

        // ---------------------------------------------------------------- Internals

        private void DrawQuad( OpenGLTexture2D texture, Vector2 p, Vector2 s, Vector2 uv0, Vector2 uv1, Color color ) {
            SetTexture( texture );
            Reserve( 6 );
            Vector2 p00 = p, p10 = new( p.X + s.X, p.Y ), p11 = p + s, p01 = new( p.X, p.Y + s.Y );
            Vector2 t00 = uv0, t10 = new( uv1.X, uv0.Y ), t11 = uv1, t01 = new( uv0.X, uv1.Y );
            PushTriangle( p00, p10, p11, t00, t10, t11, color );
            PushTriangle( p00, p11, p01, t00, t11, t01, color );
        }

        private void PushTriangle( Vector2 a, Vector2 b, Vector2 c, Vector2 ta, Vector2 tb, Vector2 tc, Color color ) {
            vertices[ vertexCount++ ] = new Vertex { Position = a, UV = ta, Color = color };
            vertices[ vertexCount++ ] = new Vertex { Position = b, UV = tb, Color = color };
            vertices[ vertexCount++ ] = new Vertex { Position = c, UV = tc, Color = color };
        }

        private void SetTexture( OpenGLTexture2D texture ) {
            if ( texture != currentTexture ) {
                Flush();
                currentTexture = texture;
            }
        }

        private void Reserve( int count ) {
            if ( count > MaxVertices ) throw new ArgumentException( $"Single draw needs {count} vertices, batch holds {MaxVertices}" );
            if ( vertexCount + count > MaxVertices ) Flush();
        }

        private unsafe void Flush() {
            if ( vertexCount == 0 ) return;

            Vector2 screen = window.Size;
            Matrix4x4 view = camera?.ViewMatrix( screen ) ?? Matrix4x4.Identity;
            // System.Numerics is row-vector/row-major, which is the same memory layout GLSL expects for
            // column-vector/column-major, so no transpose is needed.
            Matrix4x4 viewProj = view * Matrix4x4.CreateOrthographicOffCenter( 0, screen.X, screen.Y, 0, -1, 1 );

            gl.UseProgram( program );
            gl.UniformMatrix4( viewProjLocation, 1, false, (float*)&viewProj );

            gl.ActiveTexture( TextureUnit.Texture0 );
            gl.BindTexture( TextureTarget.Texture2D, currentTexture.Handle );

            gl.BindVertexArray( vao );
            gl.BindBuffer( BufferTargetARB.ArrayBuffer, vbo );
            fixed ( Vertex* data = vertices ) {
                gl.BufferSubData( BufferTargetARB.ArrayBuffer, 0, (nuint)( vertexCount * sizeof( Vertex ) ), data );
            }
            gl.DrawArrays( PrimitiveType.Triangles, 0, (uint)vertexCount );

            vertexCount = 0;
            drawCalls++;
        }

        private uint CreateProgram( string vertexSource, string fragmentSource ) {
            uint vs = CompileShader( ShaderType.VertexShader, vertexSource );
            uint fs = CompileShader( ShaderType.FragmentShader, fragmentSource );

            uint prog = gl.CreateProgram();
            gl.AttachShader( prog, vs );
            gl.AttachShader( prog, fs );
            gl.LinkProgram( prog );
            gl.GetProgram( prog, ProgramPropertyARB.LinkStatus, out int linked );
            if ( linked == 0 ) {
                throw new InvalidOperationException( "Shader link failed: " + gl.GetProgramInfoLog( prog ) );
            }

            gl.DetachShader( prog, vs );
            gl.DetachShader( prog, fs );
            gl.DeleteShader( vs );
            gl.DeleteShader( fs );
            return prog;
        }

        private uint CompileShader( ShaderType type, string source ) {
            uint shader = gl.CreateShader( type );
            gl.ShaderSource( shader, source );
            gl.CompileShader( shader );
            gl.GetShader( shader, ShaderParameterName.CompileStatus, out int compiled );
            if ( compiled == 0 ) {
                throw new InvalidOperationException( $"{type} compile failed: " + gl.GetShaderInfoLog( shader ) );
            }
            return shader;
        }

        public void Dispose() {
            whiteTexture.Dispose();
            gl.DeleteBuffer( vbo );
            gl.DeleteVertexArray( vao );
            gl.DeleteProgram( program );
        }
    }
}
