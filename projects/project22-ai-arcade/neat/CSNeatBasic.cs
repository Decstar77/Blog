namespace arcade {
    /*
    ===============================================================================
    Evolves NEAT genomes to play CS Lite as the team-0 player against a CSBot.
    Inputs are the CSGameState features (same as REINFORCE), outputs are the same
    14 action rows split into three heads (move 9, rotate 3, fire 2). NEAT outputs
    are sigmoids, so each head picks its argmax: the policy is deterministic.
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

        private readonly CSMap          map;
        private readonly int?           seed;
        private readonly int            gamesPerGenome;
        private readonly Random         seedRng;
        private readonly int[]          validationSeeds;
        private readonly List<CSPlayerAction> actions = new List<CSPlayerAction>();

        public static int               ValidationGames = 100;  // Fixed games the elites are re-scored on
        public static int               ValidateTopK    = 3;    // How many of each generation's top genomes get re-scored

        public NeatPopulation           Population { get; }
        public Genome?                  Champion { get; private set; }      // Best genome by validated win rate
        public CSTrainingLog?           ChampionLog { get; private set; }
        public Genome?                  Best => Champion;

        /*
        ==========================
        ==========================
         */
        public CSNeatBasic( CSMap map, int? seed = null, int gamesPerGenome = 8, NeatSettings? settings = null ) {
            this.map = map;
            this.seed = seed;
            this.gamesPerGenome = gamesPerGenome;
            if ( seed != null ) RNG.Seed( seed.Value );
            seedRng = seed != null ? new Random( seed.Value * 7919 + 1 ) : new Random();
            // Same validation games every run, kept away from the training seed range.
            validationSeeds = Enumerable.Range( 0, ValidationGames ).Select( i => 1_000_000 + i ).ToArray();
            Population = new NeatPopulation( CSGameState.FeatureCount, OutputCount, settings );
        }

        /*
        ==========================
        Runs evaluate + evolve for the given number of generations. Every genome in a generation
        plays the same freshly drawn games, then the top few are re-scored on a fixed validation
        set. The returned champion is the best by validated win rate, not the luckiest training score.
        ==========================
         */
        public Genome Train( int generations, int LogEvery = 1 ) {
            var timer = System.Diagnostics.Stopwatch.StartNew();
            TrainingDashboard.BeginRun( "neat-cs", new {
                generations, seed, gamesPerGenome, ValidationGames, ValidateTopK, ActionRepeat,
                inputs = CSGameState.FeatureCount, outputs = OutputCount, settings = Population.Settings
            } );
            for ( int g = 0; g < generations; g++ ) {
                double genStart = timer.Elapsed.TotalSeconds;
                int[] seeds = Enumerable.Range( 0, gamesPerGenome ).Select( _ => seedRng.Next() ).ToArray();
                var log = new CSTrainingLog();
                Population.Evaluate( genome => Fitness( genome, seeds, log ) );
                if ( ValidateElites() ) LogChampionNetwork( g );
                LogToDashboard( g, log, timer.Elapsed.TotalSeconds - genStart );

                if ( LogEvery > 0 && ( g % LogEvery == 0 || g == generations - 1 ) ) {
                    Genome best = Population.Genomes.MaxBy( x => x.fitness )!;
                    int hidden = Champion!.nodes.Count( n => n.type == GeneNodeType.Hidden );
                    int links  = Champion.connections.Count( c => c.enabled );
                    CSTrainingLog c = ChampionLog!;
                    Console.Write( $"gen {g,4}  species {Population.SpeciesCount,3} (thr {Population.CompatibilityThreshold,5:F2})  best {best.fitness,6:F2}  " +
                                   $"CHAMP win {c.WinRate,4:P0} loss {c.LossRate,4:P0} draw {c.DrawRate,4:P0} score {c.AvgReward,5:F2} hidden {hidden,3} links {links,4}  | pop " );
                    log.Print( g, generations, timer.Elapsed.TotalSeconds );
                }

                if ( g < generations - 1 ) Population.Evolve();
            }
            TrainingDashboard.EndRun();
            return Champion!;
        }

        /*
        ==========================
        One line per generation. Series sharing a prefix share a chart, so each prefix
        only holds values on the same scale.
        ==========================
         */
        private void LogToDashboard( int generation, CSTrainingLog pop, double seconds ) {
            CSTrainingLog champ = ChampionLog!;
            var fitness = Population.Genomes.Select( x => x.fitness ).ToArray();
            TrainingDashboard.Log( generation, new Dictionary<string, double> {
                ["champion win rate/win"]   = champ.WinRate,
                ["champion win rate/loss"]  = champ.LossRate,
                ["champion win rate/draw"]  = champ.DrawRate,
                ["population win rate/win"]  = pop.WinRate,
                ["population win rate/loss"] = pop.LossRate,
                ["population win rate/draw"] = pop.DrawRate,
                ["fitness/best"]            = fitness.Max(),
                ["fitness/mean"]            = fitness.Average(),
                ["fitness/champion (validation)"] = champ.AvgReward,
                ["hits per game/dealt (pop)"]  = pop.AvgDealt,
                ["hits per game/taken (pop)"]  = pop.AvgTaken,
                ["hits per game/dealt (champ)"] = champ.AvgDealt,
                ["hits per game/taken (champ)"] = champ.AvgTaken,
                ["accuracy/population"]     = pop.Accuracy,
                ["accuracy/champion"]       = champ.Accuracy,
                ["game length (s)/population"] = pop.AvgLengthSeconds,
                ["game length (s)/champion"]   = champ.AvgLengthSeconds,
                ["species/count"]           = Population.SpeciesCount,
                ["compatibility threshold/value"] = Population.CompatibilityThreshold,
                ["champion hidden nodes/count"]   = Champion!.nodes.Count( n => n.type == GeneNodeType.Hidden ),
                ["champion enabled links/count"]  = Champion.connections.Count( c => c.enabled ),
                ["seconds per generation/value"]  = seconds,
            } );
        }

        /*
        ==========================
        Re-score this generation's top genomes on the fixed validation games and keep the best.
        ==========================
         */
        private bool ValidateElites() {
            bool changed = false;
            foreach ( Genome genome in Population.Genomes.OrderByDescending( x => x.fitness ).Take( ValidateTopK ) ) {
                var log = new CSTrainingLog();
                Fitness( genome, validationSeeds, log );
                bool better = ChampionLog == null
                    || log.WinRate > ChampionLog.WinRate
                    || ( log.WinRate == ChampionLog.WinRate && log.AvgReward > ChampionLog.AvgReward );
                if ( better ) {
                    Champion = genome.Clone();
                    ChampionLog = log;
                    changed = true;
                }
            }
            return changed;
        }

        /*
        ==========================
        Champion snapshot for the dashboard's Network tab, logged only when it changes.
        Input/output nodes are labelled by their position among nodes of that type,
        which matches the order Forward reads inputs and writes outputs.
        ==========================
         */
        private void LogChampionNetwork( int generation ) {
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
                winRate = Math.Round( ChampionLog!.WinRate, 3 ),
                score   = Math.Round( ChampionLog.AvgReward, 3 ),
                hidden  = c.nodes.Count( n => n.type == GeneNodeType.Hidden ),
                links   = c.connections.Count( e => e.enabled ),
            } );
        }

        /*
        ==========================
        Average shaped score over the given games. Kept positive because NeatPopulation's
        fitness sharing and roulette selection both divide by the max fitness.
        ==========================
         */
        private double Fitness( Genome genome, int[] gameSeeds, CSTrainingLog log ) {
            double total = 0;
            foreach ( int gameSeed in gameSeeds ) {
                total += PlayGame( genome, gameSeed, log );
            }
            return Math.Max( 0, total / gameSeeds.Length );
        }

        /*
        ==========================
        ==========================
         */
        private double PlayGame( Genome genome, int? gameSeed, CSTrainingLog log ) {
            CSLite game = new CSLite( map, gameSeed );
            int agentId = game.Players.FindIndex( p => p.Team == 0 );
            int enemyId = game.Players.FindIndex( p => p.Team == 1 );
            CSBot bot = new CSBot( enemyId, gameSeed );

            int dealt = 0, taken = 0, fired = 0, ticks = 0;
            while ( game.Result == CSGameResult.PLAYING ) {
                CSPlayerAction policyAction = Think( genome, CSGameState.Create( game, agentId, enemyId ) );
                policyAction.PlayerId = agentId;

                for ( int k = 0; k < ActionRepeat && game.Result == CSGameResult.PLAYING; k++ ) {
                    actions.Clear();
                    actions.Add( bot.Think( game, (float)CSSettings.InvGameTickRate ) );
                    actions.Add( policyAction );
                    game.Step( actions );
                    ticks++;

                    foreach ( CSEvent e in game.Events ) {
                        bool hit = e.Type == CSEventType.Hit || e.Type == CSEventType.Death;
                        if ( hit && e.PlayerId == agentId ) taken++;
                        if ( hit && e.PlayerId == enemyId ) dealt++;
                        if ( e.Type == CSEventType.Shot && e.PlayerId == agentId ) fired++;
                    }
                }
            }

            // Baseline of 2 keeps an ordinary loss (4 hits taken) above zero so selection still
            // sees a gradient. Hits dealt matter most and a win adds a big step. The survival
            // bonus only applies to losses (dying late beats dying fast), so hiding until the
            // timer runs out scores below a loss where the agent actually fought.
            double score = 2.0 + 1.0 * dealt - 0.25 * taken;
            if ( game.Result == CSGameResult.LOSS ) score += ticks * CSSettings.InvGameTickRate / CSSettings.MaxRoundTime;
            if ( game.Result == CSGameResult.WIN ) score += 4.0;

            log.Add( game.Result, score, ticks, dealt, taken, fired );
            return score;
        }

        /*
        ==========================
        ==========================
         */
        public static CSPlayerAction Think( Genome genome, CSGameState s ) {
            double[] y = genome.Forward( s.X );
            int ma = ArgMax( y, 0, 9 );
            int ra = ArgMax( y, 9, 3 );
            int fa = ArgMax( y, 12, 2 );

            CSPlayerAction action = new CSPlayerAction();
            action.MoveLeft  = CSGameState.Moves[ma].L;
            action.MoveRight = CSGameState.Moves[ma].R;
            action.MoveUp    = CSGameState.Moves[ma].U;
            action.MoveDown  = CSGameState.Moves[ma].D;
            action.Rotation  = ra - 1.0f;
            action.Shoot     = fa == 0;
            return action;
        }

        private static int ArgMax( double[] y, int start, int count ) {
            int best = 0;
            for ( int i = 1; i < count; i++ ) {
                if ( y[start + i] > y[start + best] ) best = i;
            }
            return best;
        }

        /*
        ==========================
        Run a genome for the team-0 player of an external game, holding each decision
        for ActionRepeat ticks to match training.
        ==========================
         */
        private CSPlayerAction  heldAction;
        private int             heldSinceTick = int.MaxValue;

        public CSPlayerAction Act( CSLite external, Genome genome ) {
            int tick = external.TickCount;
            if ( tick < heldSinceTick || tick - heldSinceTick >= ActionRepeat ) { // tick < heldSinceTick: the game was restarted
                int me    = external.Players.FindIndex( p => p.Team == 0 );
                int enemy = external.Players.FindIndex( p => p.Team == 1 );
                heldAction = Think( genome, CSGameState.Create( external, me, enemy ) );
                heldAction.PlayerId = me;
                heldSinceTick = tick;
            }
            return heldAction;
        }
    }
}
