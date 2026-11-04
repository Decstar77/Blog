using System.Diagnostics;
using System.Numerics;
using Raylib_cs;

namespace arcade {
    // Frozen Lake trainer: pick a map and a tabular method, tune its inputs, train, then watch the policy play.
    // Training runs on a worker thread so the window stays live, the training controls lock until it finishes.
    public class FrozenLakeUI {
        private record Inputs( string[] Map, bool Slippery, int MaxSteps, int Episodes, double Alpha, double Epsilon, double Gamma, int N );

        private record Method( string Name, bool UsesEpisodes, bool UsesAlpha, bool UsesEpsilon, bool UsesN, Func<Inputs, int[]> Train );

        private static readonly Method[] Methods = {
            new( "Policy Iteration",    false, false, false, false, i => new DynamicProgramming( ModelBased( i ) ).PolicyIteration( i.Gamma ) ),
            new( "MC Exploring Starts", true,  false, false, false, i => new MonteCarloFirstVisitOnPolicy( ModelFree( i ) ).PolicyControl_ExploringStarts( i.Episodes, i.Gamma ) ),
            new( "MC Epsilon-Soft",     true,  false, true,  false, i => new MonteCarloFirstVisitOnPolicy( ModelFree( i ) ).PolicyControl_EpsilonSoft( i.Episodes, i.Epsilon, i.Gamma ) ),
            new( "MC Off-Policy",       true,  false, false, false, TrainOffPolicy ),
            new( "Sarsa",               true,  true,  true,  false, i => new Sarsa( ModelFree( i ) ).PolicyControl_OneStep( i.Episodes, i.Alpha, i.Epsilon, i.Gamma ) ),
            new( "Expected Sarsa",      true,  true,  true,  false, i => new Sarsa( ModelFree( i ) ).PolicyControl_ExpectedSarsa( i.Episodes, i.Alpha, i.Epsilon, i.Gamma ) ),
            new( "n-step Sarsa",        true,  true,  true,  true,  i => new Sarsa( ModelFree( i ) ).PolicyControl_NStep( i.Episodes, i.N, i.Alpha, i.Epsilon, i.Gamma ) ),
            new( "Q-Learning",          true,  true,  true,  false, i => new QLearning( ModelFree( i ) ).PolicyControl_OneStep( i.Episodes, i.Alpha, i.Epsilon, i.Gamma ) ),
            new( "Double Q-Learning",   true,  true,  true,  false, i => new QLearning( ModelFree( i ) ).PolicyControl_DoubleQ( i.Episodes, i.Alpha, i.Epsilon, i.Gamma ) ),
            new( "n-step Q-Learning",   true,  true,  true,  true,  i => new QLearning( ModelFree( i ) ).PolicyControl_NStep( i.Episodes, i.N, i.Alpha, i.Epsilon, i.Gamma ) ),
        };

        private static readonly string[]   MapNames = { "4x4", "8x8" };
        private static readonly string[][] Maps = { FrozenLake.Map4x4, FrozenLake.Map8x8 };
        private static readonly string[]   MethodNames = Methods.Select( m => m.Name ).ToArray();
        private static readonly int[]      EpisodeCounts = { 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000 };

        // Same order as the actions: left, down, right, up.
        private static readonly int[] DRow = { 0, 1, 0, -1 };
        private static readonly int[] DCol = { -1, 0, 1, 0 };

        private const float SideWidth = 320;
        private const float ChartHeight = 200;
        private const float Gap = 16;

        // Environment
        private int     mapIndex = 0;
        private bool    slippery = true;

        // Method and its inputs
        private int     methodIndex = 7;
        private int     episodeIndex = 6;
        private double  alpha = 0.1;
        private double  epsilon = 0.1;
        private double  gamma = 0.99;
        private int     n = 4;

        // Training result. No policy means the agent moves at random.
        private Task<(int[] Policy, double[] Values)>? training;
        private readonly Stopwatch trainTimer = new();
        private int[]?      policy;
        private double[]?   values;
        private string      trainedWith = "Untrained, moving at random";
        private string?     error;

        // Playback
        private FrozenLake  lake;
        private bool        playing = true;
        private int         stepsPerSecond = 4;
        private float       timer = 0;
        private int         wins = 0;
        private int         episodes = 0;
        private readonly List<double> winRate = new();
        private bool        showValues = true;
        private bool        showPolicy = true;

        public FrozenLakeUI() {
            lake = NewLake();
        }

        private string[] Map => Maps[mapIndex];
        private int MaxSteps => mapIndex == 0 ? 100 : 200;

        public void Update( float dt ) {
            if ( training != null && training.IsCompleted ) {
                FinishTraining();
            }

            if ( !playing ) {
                return;
            }

            float delay = 1f / stepsPerSecond;
            timer += dt;
            // Several steps a frame at high speeds, capped so a slow frame doesn't snowball.
            for ( int i = 0; timer >= delay && i < 1000; i++ ) {
                timer -= delay;
                Step();
            }
            timer = MathF.Min( timer, delay );
        }

        public void Draw( Rectangle area ) {
            UI.Begin();

            Rectangle inner = Style.Inflate( area, -Gap );
            var left = new Rectangle( inner.X, inner.Y, SideWidth, inner.Height );
            var right = new Rectangle( inner.X + inner.Width - SideWidth, inner.Y, SideWidth, inner.Height );
            var center = new Rectangle( left.X + SideWidth + Gap, inner.Y, inner.Width - 2 * ( SideWidth + Gap ), inner.Height );

            DrawEnvironment( ref left );
            DrawMethods( ref left );
            DrawResults( ref left );
            DrawInputs( ref right );
            DrawPlayback( ref right );
            DrawLake( center );

            UI.End();
        }

        private void DrawEnvironment( ref Rectangle column ) {
            UI.Enabled = training == null;
            Rectangle panel = UI.Panel( UI.TakeTop( ref column, PanelHeight( 2 * UI.RowHeight, UI.RowHeight ), Gap ), "Environment" );
            bool changed = UI.ListBox( UI.TakeTop( ref panel, 2 * UI.RowHeight ), MapNames, ref mapIndex );
            changed |= UI.Toggle( UI.TakeTop( ref panel, UI.RowHeight ), "Slippery ice", ref slippery );

            // A policy learned on another lake means nothing here.
            if ( changed ) {
                policy = null;
                values = null;
                trainedWith = "Untrained, moving at random";
                error = null;
                ResetPlayback();
            }
        }

        private void DrawMethods( ref Rectangle column ) {
            UI.Enabled = training == null;
            float height = Methods.Length * UI.RowHeight;
            Rectangle panel = UI.Panel( UI.TakeTop( ref column, PanelHeight( height ), Gap ), "Method" );
            UI.ListBox( UI.TakeTop( ref panel, height ), MethodNames, ref methodIndex );
        }

        private void DrawResults( ref Rectangle column ) {
            UI.Enabled = true;
            Rectangle panel = UI.Panel( UI.TakeTop( ref column, PanelHeight( UI.TextSize, UI.TextSize, UI.TextSize, UI.TextSize ), Gap ), "Result" );

            UI.Label( UI.TakeTop( ref panel, UI.TextSize ), trainedWith );
            if ( error != null ) {
                UI.Label( UI.TakeTop( ref panel, UI.TextSize ), error, Style.P0 );
            } else if ( policy != null ) {
                UI.Label( UI.TakeTop( ref panel, UI.TextSize ), $"Trained in {trainTimer.Elapsed.TotalSeconds:0.00} s", Style.TextDim );
            }
            if ( values != null ) {
                // Exact, from policy evaluation, so it's the number the playback win rate should settle on.
                UI.Label( UI.TakeTop( ref panel, UI.TextSize ), $"Goal chance {values[StartState()]:0.0%}", Style.Highlight );
            }
            UI.Label( UI.TakeTop( ref panel, UI.TextSize ), $"Playback won {wins:N0} of {episodes:N0}", Style.TextDim );
        }

        private void DrawInputs( ref Rectangle column ) {
            bool idle = training == null;
            Method method = Methods[methodIndex];
            float h = UI.SliderHeight;
            Rectangle panel = UI.Panel( UI.TakeTop( ref column, PanelHeight( h, h, h, h, h ), Gap ), "Inputs" );

            UI.Enabled = idle && method.UsesEpisodes;
            UI.Slider( UI.TakeTop( ref panel, h ), "Episodes", ref episodeIndex, 0, EpisodeCounts.Length - 1, i => EpisodeCounts[i].ToString( "N0" ) );
            UI.Enabled = idle && method.UsesAlpha;
            UI.Slider( UI.TakeTop( ref panel, h ), "Step size (alpha)", ref alpha, 0.01, 1, "0.00" );
            UI.Enabled = idle && method.UsesEpsilon;
            UI.Slider( UI.TakeTop( ref panel, h ), "Exploration (epsilon)", ref epsilon, 0, 1, "0.00" );
            UI.Enabled = idle;
            UI.Slider( UI.TakeTop( ref panel, h ), "Discount (gamma)", ref gamma, 0.5, 1, "0.000" );
            UI.Enabled = idle && method.UsesN;
            UI.Slider( UI.TakeTop( ref panel, h ), "Steps (n)", ref n, 1, 16 );

            UI.Enabled = idle;
            string text = idle ? $"Train {method.Name}" : $"Training... {trainTimer.Elapsed.TotalSeconds:0.0} s";
            if ( UI.Button( UI.TakeTop( ref column, 44, Gap ), text ) ) {
                StartTraining();
            }
        }

        private void DrawPlayback( ref Rectangle column ) {
            UI.Enabled = true;
            Rectangle panel = UI.Panel( UI.TakeTop( ref column, PanelHeight( UI.RowHeight, UI.SliderHeight, UI.RowHeight, UI.RowHeight ), Gap ), "Playback" );

            Rectangle buttons = UI.TakeTop( ref panel, UI.RowHeight );
            float w = ( buttons.Width - 2 * 8 ) / 3;
            if ( UI.Button( new Rectangle( buttons.X, buttons.Y, w, buttons.Height ), playing ? "Pause" : "Play" ) ) {
                playing = !playing;
            }
            if ( UI.Button( new Rectangle( buttons.X + w + 8, buttons.Y, w, buttons.Height ), "Step" ) ) {
                playing = false;
                Step();
            }
            if ( UI.Button( new Rectangle( buttons.X + 2 * ( w + 8 ), buttons.Y, w, buttons.Height ), "Reset" ) ) {
                ResetPlayback();
            }

            UI.Slider( UI.TakeTop( ref panel, UI.SliderHeight ), "Steps per second", ref stepsPerSecond, 1, 240 );
            UI.Toggle( UI.TakeTop( ref panel, UI.RowHeight ), "Show goal chance", ref showValues );
            UI.Toggle( UI.TakeTop( ref panel, UI.RowHeight ), "Show policy", ref showPolicy );
        }

        private void DrawLake( Rectangle area ) {
            var lakeArea = new Rectangle( area.X, area.Y, area.Width, area.Height - ChartHeight - Gap );
            var chart = new Rectangle( area.X, area.Y + area.Height - ChartHeight, area.Width, ChartHeight );

            lake.Render( lakeArea );
            DrawOverlay( lakeArea );
            UI.LineChart( chart, winRate, "Playback win rate" );
        }

        // Goal chance in the corner of each tile and a chevron pointing the way the policy moves.
        // Kept to the tile edges so the agent in the middle stays visible.
        private void DrawOverlay( Rectangle area ) {
            if ( policy == null ) {
                return;
            }

            string[] map = Map;
            int rows = map.Length, cols = map[0].Length;
            // Matches the grid FrozenLake.Render lays out.
            Rectangle grid = Style.FitGrid( area, cols, rows, 24 );
            float cell = grid.Width / cols;
            float gap = MathF.Max( 2, cell * 0.04f );
            int textSize = (int)Math.Clamp( cell * 0.18f, 10, 20 );
            Color ink = Raylib.Fade( Style.Background, 0.75f );

            for ( int r = 0; r < rows; r++ ) {
                for ( int c = 0; c < cols; c++ ) {
                    if ( map[r][c] == 'H' || map[r][c] == 'G' ) {
                        continue;
                    }

                    var tile = new Rectangle( grid.X + c * cell + gap, grid.Y + r * cell + gap, cell - 2 * gap, cell - 2 * gap );
                    int s = r * cols + c;

                    if ( showValues && values != null ) {
                        Raylib.DrawText( values[s].ToString( "0.00" ), (int)( tile.X + gap * 2 ), (int)( tile.Y + gap * 2 ), textSize, ink );
                    }

                    if ( showPolicy ) {
                        int a = policy[s];
                        var d = new Vector2( DCol[a], DRow[a] );
                        var side = new Vector2( -d.Y, d.X );
                        float size = cell * 0.08f;
                        Vector2 tip = Style.Center( tile ) + d * tile.Width * 0.44f;
                        Vector2 back = tip - d * size * 1.4f;
                        Vector2 p = back + side * size, q = back - side * size;
                        // Pick the winding that's counter-clockwise on screen (y points down).
                        if ( Cross( p - tip, q - tip ) > 0 ) {
                            (p, q) = (q, p);
                        }
                        Raylib.DrawTriangle( tip, p, q, ink );
                    }
                }
            }
        }

        private void StartTraining() {
            Method method = Methods[methodIndex];
            var inputs = new Inputs( Map, slippery, MaxSteps, EpisodeCounts[episodeIndex], alpha, epsilon, gamma, n );

            error = null;
            trainTimer.Restart();
            training = Task.Run( () => {
                int[] p = method.Train( inputs );
                // Undiscounted, so each value is the chance of reaching the goal from that cell, ignoring the step limit.
                double[] v = new DynamicProgramming( ModelBased( inputs ) ).PolicyEvaluation( p, 1.0 );
                return (p, v);
            } );
            trainedWith = method.UsesEpisodes ? $"{method.Name}, {inputs.Episodes:N0} episodes" : method.Name;
        }

        private void FinishTraining() {
            trainTimer.Stop();
            if ( training!.IsFaulted ) {
                error = $"Failed: {training.Exception!.InnerException?.Message}";
            } else {
                (policy, values) = training.Result;
            }
            training = null;
            ResetPlayback();
        }

        private void Step() {
            if ( lake.IsTerminal() ) {
                lake = NewLake();
                return;
            }

            int[] legal = lake.LegalActions();
            lake.Apply( policy != null ? policy[lake.State] : legal[Random.Shared.Next( legal.Length )] );

            if ( lake.IsTerminal() ) {
                episodes++;
                if ( lake.Returns()[0] > 0 ) {
                    wins++;
                }
                winRate.Add( (double)wins / episodes );
                // Linger on the final position when it's slow enough to watch.
                if ( stepsPerSecond <= 20 ) {
                    timer -= 3f / stepsPerSecond;
                }
            }
        }

        private void ResetPlayback() {
            lake = NewLake();
            timer = 0;
            wins = 0;
            episodes = 0;
            winRate.Clear();
        }

        private FrozenLake NewLake() => new FrozenLake( Map, slippery, MaxSteps );

        private int StartState() {
            string[] map = Map;
            for ( int r = 0; r < map.Length; r++ ) {
                int c = map[r].IndexOf( 'S' );
                if ( c >= 0 ) {
                    return r * map[0].Length + c;
                }
            }
            return 0;
        }

        private static FrozenLakeTabularModelBased ModelBased( Inputs i ) => new FrozenLakeTabularModelBased( i.Map, i.Slippery );
        private static FrozenLakeTabularModelFree ModelFree( Inputs i ) => new FrozenLakeTabularModelFree( i.Map, i.Slippery, i.MaxSteps );

        private static int[] TrainOffPolicy( Inputs i ) {
            FrozenLakeTabularModelFree env = ModelFree( i );
            // PolicyControl fills in the greedy target policy as it goes.
            int[] policy = new int[env.NumStates];
            new MonteCarloEveryVisitOffPolicy( env ).PolicyControl( policy, i.Episodes, i.Gamma );
            return policy;
        }

        // Height of a titled UI.Panel holding these rows stacked with UI.TakeTop's default gap.
        private static float PanelHeight( params float[] rows ) =>
            2 * UI.Padding + UI.TextSize + 8 + rows.Sum() + 8 * ( rows.Length - 1 );

        private static float Cross( Vector2 u, Vector2 v ) => u.X * v.Y - u.Y * v.X;
    }
}
