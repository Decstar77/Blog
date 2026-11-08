using Raylib_cs;
using System.Numerics;

namespace arcade {
    public class Program {
        // CS Lite: player 0 is the human (WASD to move, mouse to aim, left click / space to shoot),
        // player 1 is a CSBot. R restarts the round.
        public static void Main( string[] Args ) {
            Raylib.SetConfigFlags( ConfigFlags.Msaa4xHint | ConfigFlags.ResizableWindow | ConfigFlags.VSyncHint );
            Raylib.InitWindow( 1300, 900, "AI Arcade - CS Lite" );
            Raylib.SetWindowMinSize( 640, 480 );
            Raylib.SetTargetFPS( 60 );
            Raylib.InitAudioDevice();
            CSAudio audio = new CSAudio( listenerId: 0 );

            var map = new CSMap();
            map.ParseMapFile( Path.Combine( AppContext.BaseDirectory, "maps", "cs_map001.txt" ) );

            int seed = 40;
            CSReinforce csrl = new CSReinforce( map, seed );
            Reinforce reinforce = new Reinforce();
            double[] theta = reinforce.PolicyControl( csrl, 100, 0.1, 0.9, seed );

            CSLite game = new CSLite( map );
            CSBot bot = new CSBot( 1 );
            var actions = new List<CSPlayerAction> { default, default };

            while ( !Raylib.WindowShouldClose() ) {
                float dt = Raylib.GetFrameTime();

                if ( Raylib.IsKeyPressed( KeyboardKey.R ) ) {
                    game = new CSLite( map );
                    bot = new CSBot( 1 );
                }
                
                actions[0] = csrl.Think(  // HumanAction( game, 0 );
                actions[1] = bot.Think( game, dt );
                game.Step( dt, actions );
                audio.Play( game );

                Raylib.BeginDrawing();
                Raylib.ClearBackground( Style.Background );
                game.Render( dt, actions );
                Raylib.DrawText( "WASD move  |  mouse aim  |  LMB/Space shoot  |  R restart", 12, Raylib.GetScreenHeight() - 28, 18, Style.TextDim );
                Raylib.EndDrawing();
            }

            audio.Unload();
            Raylib.CloseAudioDevice();
            Raylib.CloseWindow();
        }

        private static CSPlayerAction HumanAction( CSLite game, int playerIndex ) {
            CSPlayer me = game.Players[playerIndex];
            var action = new CSPlayerAction {
                PlayerId  = playerIndex,
                MoveLeft  = Raylib.IsKeyDown( KeyboardKey.A ),
                MoveRight = Raylib.IsKeyDown( KeyboardKey.D ),
                MoveUp    = Raylib.IsKeyDown( KeyboardKey.W ),
                MoveDown  = Raylib.IsKeyDown( KeyboardKey.S ),
                Shoot     = Raylib.IsMouseButtonDown( MouseButton.Left ) || Raylib.IsKeyDown( KeyboardKey.Space ),
            };

            // Turn toward the mouse, limited by the same turn speed the bot has.
            Vector2 mouse = Raylib.GetScreenToWorld2D( Raylib.GetMousePosition(), game.Camera );
            Vector2 toMouse = mouse - me.Position;
            if ( toMouse.LengthSquared() > 1 ) {
                float error = CSLite.WrapAngle( MathF.Atan2( toMouse.Y, toMouse.X ) - me.Orientation );
                float maxTurnPerTick = CSSettings.PlayerTurnSpeed * (float)CSSettings.InvGameTickRate;
                action.Rotation = Math.Clamp( error / maxTurnPerTick, -1, 1 );
            }
            return action;
        }
    }

}
