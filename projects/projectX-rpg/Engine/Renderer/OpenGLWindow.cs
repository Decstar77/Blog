using System.Diagnostics;
using System.Numerics;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.Windowing;

namespace Engine {
    public sealed class OpenGLWindow : IDisposable {
        public GL       GL { get; }
        public int      Width => window.FramebufferSize.X;
        public int      Height => window.FramebufferSize.Y;
        public Vector2  Size => new Vector2( Width, Height );
        public bool     ShouldClose => window.IsClosing;
        public float    DeltaTime { get; private set; }      // Seconds since the previous PollEvents
        public double   Time => clock.Elapsed.TotalSeconds;  // Seconds since the window was created

        public Vector2  MousePosition { get; private set; }  // Pixels, origin top-left
        public float    MouseScroll { get; private set; }    // Wheel ticks this frame

        public string Title {
            get => window.Title;
            set => window.Title = value;
        }

        private readonly IWindow              window;
        private readonly IInputContext        input;
        private readonly Stopwatch            clock = Stopwatch.StartNew();
        private double                        lastFrameTime;

        private readonly HashSet<Key>         keysDown = new();
        private readonly HashSet<Key>         keysPressed = new();   // Went down this frame
        private readonly HashSet<MouseButton> buttonsDown = new();
        private readonly HashSet<MouseButton> buttonsPressed = new();
        private float                         scrollAccum;

        public OpenGLWindow( string title, int width, int height, bool vsync = true ) {
            WindowOptions options = WindowOptions.Default with {
                Title = title,
                Size = new Vector2D<int>( width, height ),
                VSync = vsync,
                API = new GraphicsAPI( ContextAPI.OpenGL, ContextProfile.Core, ContextFlags.ForwardCompatible, new APIVersion( 3, 3 ) ),
            };

            window = Window.Create( options );
            window.Initialize();

            GL = window.CreateOpenGL();
            OpenGLInstance.Set( GL );
            GL.Viewport( 0, 0, (uint)Width, (uint)Height );
            window.FramebufferResize += size => GL.Viewport( 0, 0, (uint)size.X, (uint)size.Y );

            input = window.CreateInput();
            foreach ( IKeyboard keyboard in input.Keyboards ) {
                keyboard.KeyDown += ( _, key, _ ) => { if ( keysDown.Add( key ) ) keysPressed.Add( key ); };
                keyboard.KeyUp += ( _, key, _ ) => keysDown.Remove( key );
            }
            foreach ( IMouse mouse in input.Mice ) {
                mouse.MouseDown += ( _, b ) => { if ( buttonsDown.Add( b ) ) buttonsPressed.Add( b ); };
                mouse.MouseUp += ( _, b ) => buttonsDown.Remove( b );
                mouse.MouseMove += ( _, p ) => MousePosition = p;
                mouse.Scroll += ( _, w ) => scrollAccum += w.Y;
            }

            lastFrameTime = Time;
        }

        // Pumps OS events and updates input + DeltaTime. Call once at the start of each frame.
        public void PollEvents() {
            keysPressed.Clear();
            buttonsPressed.Clear();
            scrollAccum = 0;

            window.DoEvents();

            MouseScroll = scrollAccum;
            double now = Time;
            DeltaTime = (float)( now - lastFrameTime );
            lastFrameTime = now;
        }

        public void SwapBuffers() => window.SwapBuffers();
        public void Close() => window.Close();

        public bool IsKeyDown( Key key ) => keysDown.Contains( key );
        public bool IsKeyPressed( Key key ) => keysPressed.Contains( key );
        public bool IsMouseDown( MouseButton button ) => buttonsDown.Contains( button );
        public bool IsMousePressed( MouseButton button ) => buttonsPressed.Contains( button );

        public void Dispose() {
            input.Dispose();
            GL.Dispose();
            window.Reset();
            window.Dispose();
        }
    }
}
