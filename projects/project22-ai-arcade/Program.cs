using Raylib_cs;
using System.Numerics;

namespace arcade {
    public class Program {
        // CS Lite: the team-0 player is driven by the best NEAT genome evolved at startup,
        // the team-1 player is a CSBot. R restarts the round.
        public static void Main( string[] Args ) {
            // Headless training run for tuning: dotnet run -- train [episodes] [alpha] [gamma] [seed]
            if ( Args.Length > 0 && Args[0] == "train" ) {
                Train( Args );
                return;
            }
            // Dashboard only, to browse old runs: dotnet run -- dash
            if ( Args.Length > 0 && Args[0] == "dash" ) {
                TrainingDashboard.Start();
                Console.WriteLine( "Ctrl+C to stop." );
                Thread.Sleep( Timeout.Infinite );
                return;
            }
            // Score a saved champion against CSBot on the fixed benchmark games: dotnet run -- eval [path]
            if ( Args.Length > 0 && Args[0] == "eval" ) {
                string? path = Args.Length > 1 ? Args[1] : TrainingDashboard.LatestArtifact( "champion.bin" );
                if ( path == null || !File.Exists( path ) ) {
                    Console.WriteLine( $"No champion found. Looked in {TrainingDashboard.RunsDir}" );
                    return;
                }
                var evalMap = new CSMap();
                evalMap.ParseMapFile( Path.Combine( AppContext.BaseDirectory, "maps", "cs_map001.txt" ) );
                Genome genome = Genome.Load( path );
                CSTrainingLog log = new CSNeatBasic( evalMap, settings: new NeatSettings { PopulationSize = 1 } ).BenchmarkVsBot( genome );
                Console.WriteLine( $"{Path.GetFileName( path )} vs CSBot: win {log.WinRate:P0}  loss {log.LossRate:P0}  draw {log.DrawRate:P0}  score {log.AvgReward:F3}  ({log.Episodes} games)" );
                return;
            }
            // Headless NEAT run: dotnet run -- neat [generations] [population] [seed]
            if ( Args.Length > 0 && Args[0] == "neat" ) {
                TrainingDashboard.Start();
                TrainNeat( Args );
                return;
            }

            var map = new CSMap();
            map.ParseMapFile( Path.Combine( AppContext.BaseDirectory, "maps", "cs_map001.txt" ) );

            Genome champion;
            // Play a saved champion instead of training: dotnet run -- play [path]
            // With no path it picks the newest champion.bin in diagnostics/runs.
            if ( Args.Length > 0 && Args[0] == "play" ) {
                string? path = Args.Length > 1 ? Args[1] : TrainingDashboard.LatestArtifact( "champion.bin" );
                if ( path == null || !File.Exists( path ) ) {
                    Console.WriteLine( $"No champion to play. Train first, or pass a path to a .champion.bin file. Looked in {TrainingDashboard.RunsDir}" );
                    return;
                }
                Console.WriteLine( $"Playing {path}" );
                champion = Genome.Load( path );
            }
            else {
                TrainingDashboard.Start();
                Console.WriteLine( "Training..." );

                NeatSettings neatSettings = new NeatSettings() {
                    PopulationSize = 250,
                    CompatibilityThreshold = 0.85,  // Starting value, adapts toward TargetSpeciesCount
                    TargetSpeciesCount = 15,
                    MaxStagnation = 100
                };

                int? seed = null;// 40;
                CSNeatBasic neat = new CSNeatBasic( map, seed, 8, neatSettings );
                champion = neat.Train( 250, 25 );
            }

            Raylib.SetConfigFlags( ConfigFlags.Msaa4xHint | ConfigFlags.ResizableWindow | ConfigFlags.VSyncHint );
            Raylib.InitWindow( 1300, 900, "AI Arcade - CS Lite" );
            Raylib.SetWindowMinSize( 640, 480 );
            Raylib.SetTargetFPS( 60 );
            Raylib.InitAudioDevice();
            CSAudio audio = new CSAudio( listenerId: 0 );

            CSLite game = new CSLite( map );
            int agentId = game.Players.FindIndex( p => p.Team == 0 );
            int otherId = game.Players.FindIndex( p => p.Team == 1 );
            CSBot bot = new CSBot( otherId );
            var agent  = new CSNeatController( champion, agentId );
            var mirror = new CSNeatController( champion, otherId );   // the champion playing itself
            string[] opponents = { "You", "CSBot", "Champion (self-play)" };
            int opponent = 0;
            var actions = new List<CSPlayerAction> { default, default };

            while ( !Raylib.WindowShouldClose() ) {
                float dt = Raylib.GetFrameTime();

                bool cycle = Raylib.IsKeyPressed( KeyboardKey.O );
                if ( cycle ) opponent = ( opponent + 1 ) % opponents.Length;
                if ( cycle || Raylib.IsKeyPressed( KeyboardKey.R ) ) {
                    game = new CSLite( map );
                    bot = new CSBot( otherId );
                }

                actions[0] = agent.Act( game );
                actions[1] = opponent switch {
                    0 => HumanAction( game, otherId ),
                    1 => bot.Think( game, dt ),
                    _ => mirror.Act( game ),
                };
                game.Step( dt, actions );
                audio.Play( game );

                Raylib.BeginDrawing();
                Raylib.ClearBackground( Style.Background );
                game.Render( dt, actions );
                Raylib.DrawText( $"NEAT champion vs {opponents[opponent]}  |  O change opponent  |  R restart", 12, Raylib.GetScreenHeight() - 28, 18, Style.TextDim );
                Raylib.EndDrawing();
            }

            audio.Unload();
            Raylib.CloseAudioDevice();
            Raylib.CloseWindow();
        }

        private static void Train( string[] args ) {
            var inv      = System.Globalization.CultureInfo.InvariantCulture;
            int episodes = args.Length > 1 ? int.Parse( args[1] ) : 10000;
            double alpha = args.Length > 2 ? double.Parse( args[2], inv ) : 0.3;
            double gamma = args.Length > 3 ? double.Parse( args[3], inv ) : 0.995;
            int seed     = args.Length > 4 ? int.Parse( args[4] ) : 40;

            var map = new CSMap();
            map.ParseMapFile( Path.Combine( AppContext.BaseDirectory, "maps", "cs_map001.txt" ) );
            Console.WriteLine( $"REINFORCE  episodes={episodes}  alpha={alpha}  gamma={gamma}  seed={seed}" );
            new Reinforce().PolicyControl( new CSReinforceLinear( map, seed ), episodes, alpha, gamma, seed, LogEvery: Math.Max( 1, episodes / 20 ) );
        }

        private static void TrainNeat( string[] args ) {
            int generations = args.Length > 1 ? int.Parse( args[1] ) : 50;
            int population  = args.Length > 2 ? int.Parse( args[2] ) : 150;
            int seed        = args.Length > 3 ? int.Parse( args[3] ) : 40;

            var map = new CSMap();
            map.ParseMapFile( Path.Combine( AppContext.BaseDirectory, "maps", "cs_map001.txt" ) );
            Console.WriteLine( $"NEAT  generations={generations}  population={population}  seed={seed}" );
            new CSNeatBasic( map, seed, settings: new NeatSettings { PopulationSize = population } ).Train( generations );
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
