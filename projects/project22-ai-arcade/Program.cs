using Raylib_cs;
using System.Numerics;

namespace arcade {
    public class Program {
        // CS Lite: the team-0 player is driven by the REINFORCE policy trained at startup,
        // the team-1 player is a CSBot. R restarts the round.
        public static void Main( string[] Args ) {
            Console.WriteLine( "Training..." );

            var map = new CSMap();
            map.ParseMapFile( Path.Combine( AppContext.BaseDirectory, "maps", "cs_map001.txt" ) );

            int seed = 40;
            CSReinforce csrl = new CSReinforce( map, seed );
            Reinforce reinforce = new Reinforce();
            double[] theta = reinforce.PolicyControl( csrl, 10000, 0.1, 0.9, seed );

            Raylib.SetConfigFlags( ConfigFlags.Msaa4xHint | ConfigFlags.ResizableWindow | ConfigFlags.VSyncHint );
            Raylib.InitWindow( 1300, 900, "AI Arcade - CS Lite" );
            Raylib.SetWindowMinSize( 640, 480 );
            Raylib.SetTargetFPS( 60 );
            Raylib.InitAudioDevice();
            CSAudio audio = new CSAudio( listenerId: 0 );

            CSLite game = new CSLite( map );
            int botId = game.Players.FindIndex( p => p.Team == 1 );
            CSBot bot = new CSBot( botId );
            var actions = new List<CSPlayerAction> { default, default };

            while ( !Raylib.WindowShouldClose() ) {
                float dt = Raylib.GetFrameTime();

                if ( Raylib.IsKeyPressed( KeyboardKey.R ) ) {
                    game = new CSLite( map );
                    bot = new CSBot( botId );
                }
                
                actions[0] = csrl.Act( game, theta );  // or HumanAction( game, 0 ) to play yourself
                actions[1] = bot.Think( game, dt );
                game.Step( dt, actions );
                audio.Play( game );

                Raylib.BeginDrawing();
                Raylib.ClearBackground( Style.Background );
                game.Render( dt, actions );
                Raylib.DrawText( "REINFORCE agent vs CSBot  |  R restart", 12, Raylib.GetScreenHeight() - 28, 18, Style.TextDim );
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
