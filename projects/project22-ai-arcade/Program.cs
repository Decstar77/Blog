using Raylib_cs;

namespace arcade {
    public class Program {
        public static void Main( string[] Args ) {
            Raylib.SetConfigFlags( ConfigFlags.Msaa4xHint | ConfigFlags.ResizableWindow | ConfigFlags.VSyncHint );
            Raylib.InitWindow( 1400, 860, "AI Arcade - Frozen Lake" );
            Raylib.SetWindowMinSize( 1100, 860 );
            Raylib.SetTargetFPS( 60 );

            FrozenLakeUI ui = new FrozenLakeUI();
            while ( !Raylib.WindowShouldClose() ) {
                ui.Update( Raylib.GetFrameTime() );

                Raylib.BeginDrawing();
                Raylib.ClearBackground( Style.Background );
                ui.Draw( new Rectangle( 0, 0, Raylib.GetScreenWidth(), Raylib.GetScreenHeight() ) );
                Raylib.EndDrawing();
            }

            Raylib.CloseWindow();
        }
    }

}
