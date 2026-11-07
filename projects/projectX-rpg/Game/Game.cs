using System.Numerics;
using Engine;
using Engine.Resources;

namespace game {
    public static class Game {
        public static void Main( string[] args ) {
            using OpenGLWindow window = new OpenGLWindow( "projectX-rpg", 1280, 720 );
            using OpenGLRenderer renderer = new OpenGLRenderer( window );
            ResourceManager.Intialize( "C:/Projects/2025/Blog/projects/projectX-rpg" );

            while ( !window.ShouldClose ) {
                window.PollEvents();

                window.SwapBuffers();
            }
        }
    }
}
