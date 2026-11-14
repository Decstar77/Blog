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

        private readonly CSMap          map;
        private readonly int?           seed;
        private readonly int            gamesPerGenome;
        private readonly List<CSPlayerAction> actions = new List<CSPlayerAction>();

        public NeatPopulation           Population { get; }
        public Genome?                  Best => Population.BestGenome;

        /*
        ==========================
        ==========================
         */
        public CSNeatBasic( CSMap map, int? seed = null, int gamesPerGenome = 3, NeatSettings? settings = null ) {
            this.map = map;
            this.seed = seed;
            this.gamesPerGenome = gamesPerGenome;
            if ( seed != null ) RNG.Seed( seed.Value );
            Population = new NeatPopulation( CSGameState.FeatureCount, OutputCount, settings );
        }

        /*
        ==========================
        Runs evaluate + evolve for the given number of generations, returns the best genome seen.
        ==========================
         */
        public Genome Train( int generations, int LogEvery = 1 ) {
            var timer = System.Diagnostics.Stopwatch.StartNew();
            for ( int g = 0; g < generations; g++ ) {
                var log = new CSTrainingLog();
                Population.Evaluate( genome => Fitness( genome, log ) );

                if ( LogEvery > 0 && ( g % LogEvery == 0 || g == generations - 1 ) ) {
                    Genome best = Population.Genomes.MaxBy( x => x.fitness )!;
                    int hidden = best.nodes.Count( n => n.type == GeneNodeType.Hidden );
                    int links  = best.connections.Count( c => c.enabled );
                    Console.Write( $"gen {g,4}  species {Population.SpeciesCount,3}  best {best.fitness,6:F2}  (allTime {Population.BestGenome!.fitness,6:F2})  hidden {hidden,3}  links {links,4}  | " );
                    log.Print( g, generations, timer.Elapsed.TotalSeconds );
                }

                if ( g < generations - 1 ) Population.Evolve();
            }
            return Population.BestGenome!;
        }

        /*
        ==========================
        Average shaped score over a few games. Kept positive because NeatPopulation's
        fitness sharing and roulette selection both divide by the max fitness.
        ==========================
         */
        private double Fitness( Genome genome, CSTrainingLog log ) {
            double total = 0;
            for ( int i = 0; i < gamesPerGenome; i++ ) {
                int? gameSeed = seed != null ? seed.Value + i : null;
                total += PlayGame( genome, gameSeed, log );
            }
            return Math.Max( 0, total / gamesPerGenome );
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
            // sees a gradient. Hits dealt matter most, a win adds a big step, and surviving
            // longer beats dying fast.
            double survived = ticks * CSSettings.InvGameTickRate / CSSettings.MaxRoundTime;
            double score = 2.0 + 1.0 * dealt - 0.25 * taken + 1.0 * survived;
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
