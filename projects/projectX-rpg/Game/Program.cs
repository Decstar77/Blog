using System.Numerics;
using Engine;
using Engine.Resources;

namespace game {
    public static class Program {
        public static void Main( string[] args ) {
            using OpenGLWindow window = new OpenGLWindow( "projectX-rpg", 1280, 720 );
            using OpenGLRenderer renderer = new OpenGLRenderer( window );
            using ResourceManager resources = new ResourceManager( "C:/Projects/2025/Blog/projects/projectX-rpg" );
            ImageResource? sky = ResourceManager.Load<ImageResource>( "Textures/FS002_Day_Sunless.png" );

            while ( !window.ShouldClose ) {
                window.PollEvents();

                renderer.BeginFrame( Color.DarkGray );
                if ( sky != null ) {
                    renderer.DrawTexture( sky.GLTexture, Vector2.Zero, window.Size );
                }
                renderer.EndFrame();

                window.SwapBuffers();
            }
        }
    }
}
