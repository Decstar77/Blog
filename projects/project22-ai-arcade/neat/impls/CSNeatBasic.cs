namespace arcade {
    /*
    ===============================================================================
    Evolves NEAT genomes to play CS Lite by self-play.

    Inputs are the CSGameState features (same as REINFORCE), outputs are the same
    14 action rows split into three heads (move 9, rotate 3, fire 2). NEAT outputs
    are sigmoids, so each head picks its argmax: the policy is deterministic.
    Team 1 sees the map rotated into team 0's frame (see CSGameState.Create), so one
    genome can play either seat.

    Each generation:
      1. Build an opponent panel: half past champions from the hall of fame, half
         genomes from the current population. Every genome plays the same panel on
         the same seeds, alternating seats, so fitness is comparable across genomes.
      2. The top few genomes play the most recent hall-of-fame champions on fixed
         seeds. The best of them becomes this generation's champion, gets an Elo
         rating from those games, and joins the hall of fame.
      3. The champion plays CSBot on fixed seeds. CSBot is never trained against, so
         this is an outside yardstick for whether self-play is learning to play well
         or just learning to beat its own quirks.
    ===============================================================================
    */
    public class CSNeatBasic {
        public static int ActionRepeat  = 6;    // Ticks each decision is held for, same as CSReinforceLinear
        public static int OutputCount   = 9 + 3 + 2;

        // Output node names in Think's order: 9 move rows, 3 rotate rows, 2 fire rows.
        public static readonly string[] OutputNames = {
            "move none", "move left", "move right", "move up", "move down", "move up-left", "move down-left", "move up-right", "move down-right",
            "turn left", "turn none", "turn right",
            "fire", "hold fire",
        };

        public static int       HallOfFameShare     = 2;    // 1 in N panel slots goes to the hall of fame (2 = half)
        public static int       ValidateTopK        = 3;    // Candidates per generation re-scored against recent champions
        public static int       RatingOpponents     = 10;   // Most recent hall-of-fame champions a candidate is rated against
        public static int       BotBenchmarkGames   = 100;  // Fixed games vs CSBot for the outside yardstick
        public static int       NetworkLogEvery     = 10;   // Network snapshot every N generations (and whenever the topology changes)
        public static double    InitialRating       = 1000;

        public sealed class HallOfFameEntry {
            public required Genome  Genome;
            public required double  Rating;
            public required int     Generation;
        }

        private readonly CSMap          map;
        private readonly int?           seed;
        private readonly int            gamesPerGenome;
        private readonly Random         rng;
        private readonly int[]          botSeeds;
        private readonly int[]          ratingSeeds;
        private readonly List<CSPlayerAction> actions = new List<CSPlayerAction> { default, default };

        public NeatPopulation               Population { get; }
        public List<HallOfFameEntry>        HallOfFame { get; } = new List<HallOfFameEntry>();
        public Genome?                      Champion => HallOfFame.Count > 0 ? HallOfFame[^1].Genome : null;

        /*
        ==========================
        ==========================
         */
        public CSNeatBasic( CSMap map, int? seed = null, int gamesPerGenome = 8, NeatSettings? settings = null ) {
            this.map = map;
            this.seed = seed;
            this.gamesPerGenome = gamesPerGenome;
            if ( seed != null ) RNG.Seed( seed.Value );
            rng = seed != null ? new Random( seed.Value * 7919 + 1 ) : new Random();
            // Fixed games, the same every run, kept away from the training seed range.
            botSeeds    = Enumerable.Range( 0, BotBenchmarkGames ).Select( i => 1_000_000 + i ).ToArray();
            ratingSeeds = Enumerable.Range( 0, RatingOpponents * 2 ).Select( i => 2_000_000 + i ).ToArray();
            Population = new NeatPopulation( CSGameState.FeatureCount, OutputCount, settings );
        }

        /*
        ==========================
        Runs evaluate + evolve for the given number of generations and returns the last
        generation's champion.
        ==========================
         */
        public Genome Train( int generations, int LogEvery = 1 ) {
            var timer = System.Diagnostics.Stopwatch.StartNew();
            TrainingDashboard.BeginRun( "neat-cs-selfplay", new {
                generations, seed, gamesPerGenome, HallOfFameShare, ValidateTopK, RatingOpponents, BotBenchmarkGames, ActionRepeat,
                inputs = CSGameState.FeatureCount, outputs = OutputCount, settings = Population.Settings
            } );

            (int hidden, int links) lastLoggedTopology = (-1, -1);
            for ( int g = 0; g < generations; g++ ) {
                double genStart = timer.Elapsed.TotalSeconds;

                // 1. Self-play fitness against a shared panel.
                List<Genome> panel = BuildPanel();
                int[] seeds = panel.Select( _ => rng.Next() ).ToArray();
                var selfPlay = new CSTrainingLog();
                Population.Evaluate( genome => SelfPlayFitness( genome, panel, seeds, selfPlay ) );

                // 2. Pick and rate this generation's champion.
                CSTrainingLog vsRecent = CrownChampion( g, out double rating );
                Genome champion = Champion!;

                // 3. Outside yardstick.
                CSTrainingLog vsBot = BenchmarkVsBot( champion );

                SaveChampion();
                var topology = ( Hidden( champion ), Links( champion ) );
                if ( g % NetworkLogEvery == 0 || g == generations - 1 || topology != lastLoggedTopology ) {
                    LogChampionNetwork( g, rating, vsBot );
                    lastLoggedTopology = topology;
                }
                LogToDashboard( g, selfPlay, vsRecent, vsBot, rating, timer.Elapsed.TotalSeconds - genStart );

                if ( LogEvery > 0 && ( g % LogEvery == 0 || g == generations - 1 ) ) {
                    double best = Population.Genomes.Max( x => x.fitness );
                    Console.WriteLine(
                        $"gen {g,4}  species {Population.SpeciesCount,3}  best {best,5:F2}  elo {rating,6:F0}  " +
                        $"vsRecent W/L/D {Pct( vsRecent.WinRate )}/{Pct( vsRecent.LossRate )}/{Pct( vsRecent.DrawRate )}  " +
                        $"vsBot win {Pct( vsBot.WinRate )}  selfplay draw {Pct( selfPlay.DrawRate )} len {selfPlay.AvgLengthSeconds,4:F1}s  " +
                        $"hidden {topology.Item1,3} links {topology.Item2,4}  [{timer.Elapsed.TotalSeconds:F0}s]" );
                }

                if ( g < generations - 1 ) Population.Evolve();
            }
            TrainingDashboard.EndRun();
            return Champion!;
        }

        private static string Pct( double v ) => $"{v * 100,3:F0}%";
        private static int Hidden( Genome g ) => g.nodes.Count( n => n.type == GeneNodeType.Hidden );
        private static int Links( Genome g ) => g.connections.Count( c => c.enabled );

        /*
        ==========================
        Opponents every genome faces this generation. Hall-of-fame slots favour recent
        champions but keep sampling old ones, so the population can't forget how to beat
        strategies it has already seen. Before there is a hall of fame, it's all population.
        ==========================
         */
        private List<Genome> BuildPanel() {
            var panel = new List<Genome>( gamesPerGenome );
            for ( int i = 0; i < gamesPerGenome; i++ ) {
                bool fromHall = HallOfFame.Count > 0 && i % HallOfFameShare == 0;
                if ( fromHall ) {
                    bool recent = ( i / HallOfFameShare ) % 2 == 0;
                    int index = recent ? HallOfFame.Count - 1 - rng.Next( Math.Min( 5, HallOfFame.Count ) ) : rng.Next( HallOfFame.Count );
                    panel.Add( HallOfFame[index].Genome );
                }
                else {
                    panel.Add( Population.Genomes[rng.Next( Population.Genomes.Count )].Clone() );
                }
            }
            return panel;
        }

        /*
        ==========================
        Average shaped score over one game per panel opponent, alternating seats. Kept
        positive because NeatPopulation's fitness sharing and roulette selection both
        divide by the max fitness.
        ==========================
         */
        private double SelfPlayFitness( Genome genome, List<Genome> panel, int[] seeds, CSTrainingLog log ) {
            double total = 0;
            for ( int i = 0; i < panel.Count; i++ ) {
                total += PlayMatch( genome, panel[i], seeds[i], agentTeam: i % 2, log ).Score;
            }
            return Math.Max( 0, total / panel.Count );
        }

        /*
        ==========================
        The top candidates play the most recent champions, once from each seat, on fixed
        seeds. The one with the most points (win 1, draw 0.5) becomes champion and is rated
        from those games. The first generation has nobody to play, so it takes the best
        fitness and the starting rating.
        ==========================
         */
        private CSTrainingLog CrownChampion( int generation, out double rating ) {
            var candidates = Population.Genomes.OrderByDescending( x => x.fitness ).Take( ValidateTopK ).ToList();
            var opponents = HallOfFame.Skip( Math.Max( 0, HallOfFame.Count - RatingOpponents ) ).ToList();

            Genome bestGenome = candidates[0];
            CSTrainingLog bestLog = new CSTrainingLog();
            var bestResults = new List<(double opponentRating, double points)>();
            double bestPoints = double.NegativeInfinity;

            if ( opponents.Count > 0 ) {
                foreach ( Genome candidate in candidates ) {
                    var log = new CSTrainingLog();
                    var results = new List<(double, double)>();
                    double points = 0;
                    for ( int j = 0; j < opponents.Count; j++ ) {
                        for ( int side = 0; side < 2; side++ ) {
                            Match m = PlayMatch( candidate, opponents[j].Genome, ratingSeeds[j * 2 + side], side, log );
                            results.Add( (opponents[j].Rating, m.Points) );
                            points += m.Points;
                        }
                    }
                    if ( points > bestPoints ) {
                        bestPoints = points;
                        bestGenome = candidate;
                        bestLog = log;
                        bestResults = results;
                    }
                }
            }

            rating = opponents.Count == 0 ? InitialRating : PerformanceRating( bestResults );
            HallOfFame.Add( new HallOfFameEntry { Genome = bestGenome.Clone(), Rating = rating, Generation = generation } );
            return bestLog;
        }

        /*
        ==========================
        Elo performance rating: the rating R at which the expected score against these
        opponents equals the actual score. One virtual draw against the average opponent
        keeps it finite when a candidate wins (or loses) every game. Ratings are relative to
        the hall of fame, so a rising curve means each champion beats the ones before it.
        ==========================
         */
        public static double PerformanceRating( List<(double opponentRating, double points)> games ) {
            if ( games.Count == 0 ) return InitialRating;
            var all = new List<(double opponentRating, double points)>( games ) { ( games.Average( x => x.opponentRating ), 0.5 ) };
            double actual = all.Sum( x => x.points );

            double lo = -10000, hi = 10000;
            for ( int it = 0; it < 100; it++ ) {
                double mid = ( lo + hi ) / 2;
                double expected = all.Sum( x => 1.0 / ( 1.0 + Math.Pow( 10, ( x.opponentRating - mid ) / 400 ) ) );
                if ( expected < actual ) lo = mid; else hi = mid;
            }
            return ( lo + hi ) / 2;
        }

        /*
        ==========================
        Fixed games against CSBot, half from each seat.
        ==========================
         */
        public CSTrainingLog BenchmarkVsBot( Genome genome ) {
            var log = new CSTrainingLog();
            for ( int i = 0; i < botSeeds.Length; i++ ) {
                PlayMatch( genome, null, botSeeds[i], agentTeam: i % 2, log );
            }
            return log;
        }

        /*
        ==========================
        One game. A null opponent means CSBot. Results are from the agent's point of view:
        WIN means the agent won, whichever team it was on.
        ==========================
         */
        public struct Match {
            public CSGameResult Result;
            public int          Ticks, Dealt, Taken, Fired;
            public double       Score;
            public double       Points => Result == CSGameResult.WIN ? 1 : Result == CSGameResult.DRAW ? 0.5 : 0;
        }

        private Match PlayMatch( Genome agent, Genome? opponent, int gameSeed, int agentTeam, CSTrainingLog log ) {
            CSLite game = new CSLite( map, gameSeed );
            int agentId = game.Players.FindIndex( p => p.Team == agentTeam );
            int enemyId = game.Players.FindIndex( p => p.Team != agentTeam );

            var me = new CSNeatController( agent, agentId );
            CSNeatController? enemyNet = opponent != null ? new CSNeatController( opponent, enemyId ) : null;
            CSBot? bot = opponent == null ? new CSBot( enemyId, gameSeed ) : null;

            Match m = new Match();
            while ( game.Result == CSGameResult.PLAYING ) {
                actions[0] = me.Act( game );
                actions[1] = enemyNet != null ? enemyNet.Act( game ) : bot!.Think( game, (float)CSSettings.InvGameTickRate );
                game.Step( actions );
                m.Ticks++;

                foreach ( CSEvent e in game.Events ) {
                    bool hit = e.Type == CSEventType.Hit || e.Type == CSEventType.Death;
                    if ( hit && e.PlayerId == agentId ) m.Taken++;
                    if ( hit && e.PlayerId == enemyId ) m.Dealt++;
                    if ( e.Type == CSEventType.Shot && e.PlayerId == agentId ) m.Fired++;
                }
            }

            // Game results are in team terms (WIN = team 0 won), flip them for a team-1 agent.
            m.Result = game.Result;
            if ( agentTeam == 1 && m.Result == CSGameResult.WIN ) m.Result = CSGameResult.LOSS;
            else if ( agentTeam == 1 && m.Result == CSGameResult.LOSS ) m.Result = CSGameResult.WIN;

            // Baseline of 2 keeps an ordinary loss (4 hits taken) above zero so selection still
            // sees a gradient. Hits dealt matter most and a win adds a big step. The survival
            // bonus only applies to losses (dying late beats dying fast), so hiding until the
            // timer runs out scores below a loss where the agent actually fought.
            m.Score = 2.0 + 1.0 * m.Dealt - 0.25 * m.Taken;
            if ( m.Result == CSGameResult.LOSS ) m.Score += m.Ticks * CSSettings.InvGameTickRate / CSSettings.MaxRoundTime;
            if ( m.Result == CSGameResult.WIN ) m.Score += 4.0;

            log.Add( m.Result, m.Score, m.Ticks, m.Dealt, m.Taken, m.Fired );
            return m;
        }

        /*
        ==========================
        One line per generation. Series sharing a prefix share a chart, so each prefix
        only holds values on the same scale. Self-play win rate is left out on purpose:
        the population plays itself, so it sits near 50% whatever happens.
        ==========================
         */
        private void LogToDashboard( int generation, CSTrainingLog selfPlay, CSTrainingLog vsRecent, CSTrainingLog vsBot, double rating, double seconds ) {
            Genome champion = Champion!;
            var fitness = Population.Genomes.Select( x => x.fitness ).ToArray();
            var values = new Dictionary<string, double> {
                ["Elo rating/champion"]                 = rating,
                ["vs CSBot (benchmark)/win"]            = vsBot.WinRate,
                ["vs CSBot (benchmark)/loss"]           = vsBot.LossRate,
                ["vs CSBot (benchmark)/draw"]           = vsBot.DrawRate,
                ["self-play draw rate/population"]      = selfPlay.DrawRate,
                ["fitness/best"]                        = fitness.Max(),
                ["fitness/mean"]                        = fitness.Average(),
                ["hits dealt per game/self-play"]       = selfPlay.AvgDealt,
                ["hits dealt per game/champion vs bot"] = vsBot.AvgDealt,
                ["hits dealt per game/bot vs champion"] = vsBot.AvgTaken,
                ["accuracy/self-play"]                  = selfPlay.Accuracy,
                ["accuracy/champion vs bot"]            = vsBot.Accuracy,
                ["game length (s)/self-play"]           = selfPlay.AvgLengthSeconds,
                ["game length (s)/champion vs bot"]     = vsBot.AvgLengthSeconds,
                ["species/count"]                       = Population.SpeciesCount,
                ["compatibility threshold/value"]       = Population.CompatibilityThreshold,
                ["champion hidden nodes/count"]         = Hidden( champion ),
                ["champion enabled links/count"]        = Links( champion ),
                ["hall of fame/size"]                   = HallOfFame.Count,
                ["seconds per generation/value"]        = seconds,
            };
            // Nothing to compare against on the first generation.
            if ( vsRecent.Episodes > 0 ) {
                values["champion vs recent champions/win"]  = vsRecent.WinRate;
                values["champion vs recent champions/loss"] = vsRecent.LossRate;
                values["champion vs recent champions/draw"] = vsRecent.DrawRate;
            }
            TrainingDashboard.Log( generation, values );
        }

        /*
        ==========================
        Overwrites runs/<run>.champion.bin with the current champion. Load it back with
        Genome.Load, or play it with `dotnet run -- play [path]`.
        ==========================
         */
        private void SaveChampion() {
            string? path = TrainingDashboard.ArtifactPath( "champion.bin" );
            if ( path != null ) Champion!.Save( path );
        }

        /*
        ==========================
        Champion snapshot for the dashboard's Network tab. Input/output nodes are labelled
        by their position among nodes of that type, which matches the order Forward reads
        inputs and writes outputs.
        ==========================
         */
        private void LogChampionNetwork( int generation, double rating, CSTrainingLog vsBot ) {
            Genome c = Champion!;
            var nodes = new List<NetNode>();
            int inputIndex = 0, outputIndex = 0;
            foreach ( GeneNode n in c.nodes ) {
                switch ( n.type ) {
                    case GeneNodeType.Input:
                        nodes.Add( new NetNode( n.nodeId, "input", n.layer, CSGameState.FeatureNames.ElementAtOrDefault( inputIndex++ ), 0 ) );
                        break;
                    case GeneNodeType.Output:
                        nodes.Add( new NetNode( n.nodeId, "output", n.layer, OutputNames.ElementAtOrDefault( outputIndex++ ), n.bias ) );
                        break;
                    default:
                        nodes.Add( new NetNode( n.nodeId, "hidden", n.layer, null, n.bias ) );
                        break;
                }
            }
            var edges = c.connections.Select( e => new NetEdge( e.src.nodeId, e.dst.nodeId, e.weight, e.enabled ) );
            TrainingDashboard.LogNetwork( generation, "champion", nodes, edges, new {
                generation,
                elo         = Math.Round( rating ),
                vsBotWin    = Math.Round( vsBot.WinRate, 3 ),
                hidden      = Hidden( c ),
                links       = Links( c ),
            } );
        }

        /*
        ==========================
        Policy for player `me`. Team 1's observation is already rotated into team 0's
        frame by CSGameState.Create, so its absolute move is rotated back here.
        ==========================
         */
        public static CSPlayerAction Think( Genome genome, CSLite game, int me, int enemy ) {
            double[] y = genome.Forward( CSGameState.Create( game, me, enemy ).X );
            int ma = ArgMax( y, 0, 9 );
            int ra = ArgMax( y, 9, 3 );
            int fa = ArgMax( y, 12, 2 );

            var move = CSGameState.MirrorMove( CSGameState.Moves[ma], game.Players[me].Team );
            return new CSPlayerAction {
                PlayerId  = me,
                MoveLeft  = move.L,
                MoveRight = move.R,
                MoveUp    = move.U,
                MoveDown  = move.D,
                Rotation  = ra - 1.0f,
                Shoot     = fa == 0,
            };
        }

        private static int ArgMax( double[] y, int start, int count ) {
            int best = 0;
            for ( int i = 1; i < count; i++ ) {
                if ( y[start + i] > y[start + best] ) best = i;
            }
            return best;
        }
    }

    /*
    ===============================================================================
    Drives one player of a CSLite game from a genome, holding each decision for
    ActionRepeat ticks to match training. Used in training and in the game window.
    ===============================================================================
    */
    public sealed class CSNeatController {
        public Genome           Genome;
        public int              PlayerId;
        private CSPlayerAction  held;
        private int             heldSinceTick = int.MaxValue;

        public CSNeatController( Genome genome, int playerId ) {
            Genome = genome;
            PlayerId = playerId;
        }

        public CSPlayerAction Act( CSLite game ) {
            int tick = game.TickCount;
            if ( tick < heldSinceTick || tick - heldSinceTick >= CSNeatBasic.ActionRepeat ) { // tick < heldSinceTick: the game was restarted
                int myTeam = game.Players[PlayerId].Team;
                int enemy = game.Players.FindIndex( p => p.Team != myTeam );
                held = CSNeatBasic.Think( Genome, game, PlayerId, enemy );
                heldSinceTick = tick;
            }
            return held;
        }
    }
}
