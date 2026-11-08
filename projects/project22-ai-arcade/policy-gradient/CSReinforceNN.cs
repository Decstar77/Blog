namespace arcade {
    public struct GradDataNN {
        public double[] results;
        public double[] m;
        public double[] r;
        public double[] f;
        public int ma;
        public int ra;
        public int fa;
    }

    public class CSReinforceNN : PolicyGradientInterface<CSGameState, CSPlayerAction, GradDataNN> {
        private CSBot                   bot;
        private CSLite                  game;
        private CSMap                   map;
        private int?                    seed;
        private Random                  rng;
        private List<CSPlayerAction>    actions;
        private int                     agentId;
        private int                     enemyId;

        private static int[] LayerInputs  = { CSGameState.FeatureCount, 50 };
        private static int[] LayerNC      = { 50, 14 };
        private static int[] LayerSizes   = { LayerInputs[0] * LayerNC[0] + LayerNC[0], LayerInputs[1] * LayerNC[1] + LayerNC[1] };
        private static int[] LayerOffsets = { 0, LayerSizes[0] };

        /*
        ==========================
        ==========================
         */
        public CSReinforceNN( CSMap map, int? seed ) {
            this.map = map;
            this.seed = seed;
            this.actions = new List<CSPlayerAction>();
            this.rng = seed != null ? new Random( seed.Value ) : new Random();
        }
       

        /*
        ==========================
        ==========================
         */
        public void NewGame() {
            game = new CSLite( map, seed );
            agentId = game.Players.FindIndex( p => p.Team == 0 );
            enemyId = game.Players.FindIndex( p => p.Team == 1 );
            bot = new CSBot( enemyId, seed );
        }

        /*
        ==========================
        ==========================
         */
        public double[] InitialTheta() { 
            double[] theta = new double[LayerSizes[0] + LayerSizes[1]];
            for ( int i = 0; i < theta.Length; i++ ) {
                theta[i] = rng.NextDouble() * 2 - 1;
            }
            return theta;
        }

        private int ThetaWeightIndexFor(int layerIndex, int neuron, int weight ) {
            return LayerOffsets[layerIndex] + neuron * LayerInputs[layerIndex] + weight;
        }

        private int ThetaBiasIndexFor( int layerIndex, int neuron ) {
            return LayerOffsets[layerIndex] + LayerInputs[layerIndex] * LayerNC[layerIndex] + neuron;
        }

        private static double[] HeadProbs( double[] x, int start, int n ) {
            double[] p = new double[n];
            double max = double.NegativeInfinity;
            for ( int r = 0; r < n; r++ ) {
                max = Math.Max( max, x[start + r] );
            }

            double sum = 0;
            for ( int r = 0; r < n; r++ ) { p[r] = Math.Exp( x[start + r] - max ); sum += p[r]; } // subtract max: no overflow
            for ( int r = 0; r < n; r++ ) p[r] /= sum;
            return p;
        }

        /*
        ==========================
        ==========================
         */
        public (CSPlayerAction, GradDataNN) Think( CSGameState s, double[] theta ) {
            List<double[]> inputs = [s.X];
            for ( int l = 0; l < LayerSizes.Length; l++ ) {
                inputs.Add( new double[LayerNC[l]] );

                for ( int n = 0; n < LayerNC[l]; n++ ) {
                    double sum = theta[ThetaBiasIndexFor(l, n)];
                    for ( int w = 0; w < LayerInputs[l]; w++ ) {
                        int index = ThetaWeightIndexFor(l, n, w);
                        sum += theta[index] * inputs[l][w];
                    }

                    inputs[l + 1][n] = l < LayerSizes.Length - 1 ? ActivationFuncs.Relu( sum ) : sum;
                }
            }

            GradDataNN grad = new GradDataNN();
            grad.results = inputs.Last();
            grad.m = HeadProbs( grad.results, 0, 9 );
            grad.r = HeadProbs( grad.results, 9, 3 );
            grad.f = HeadProbs( grad.results, 12, 2 );
            grad.ma = Helpers.Choose( grad.m.Length, i => grad.m[i], rng );
            grad.ra = Helpers.Choose( grad.r.Length, i => grad.r[i], rng );
            grad.fa = Helpers.Choose( grad.f.Length, i => grad.f[i], rng );

            CSPlayerAction action = new CSPlayerAction();
            action.PlayerId  = agentId;
            action.MoveLeft  = CSGameState.Moves[grad.ma].L;
            action.MoveRight = CSGameState.Moves[grad.ma].R;
            action.MoveUp    = CSGameState.Moves[grad.ma].U;
            action.MoveDown  = CSGameState.Moves[grad.ma].D;
            action.Rotation  = (float)grad.ra - 1.0f;
            action.Shoot     = grad.fa == 0;

            return (action, grad);
        }

        /*
        ==========================
        ==========================
         */
        public void AccumulateGradLogProb( CSGameState s, CSPlayerAction a, GradDataNN cache, double[] theta, double scale, double[] into ) {
            throw new NotImplementedException();
        }

        /*
        ==========================
        ==========================
         */
        public float Step( CSPlayerAction policyAction ) {
            CSPlayerAction botAction = bot.Think( game, (float)CSSettings.InvGameTickRate );
            actions.Clear();
            actions.Add( botAction );
            actions.Add( policyAction );
            game.Step( actions );
            //Ticks++;

            float reward = 0;
            if ( game.Result == CSGameResult.WIN ) {
                reward = 1;
            }
            else if ( game.Result == CSGameResult.LOSS ) {
                reward = -1;
            }

            foreach ( CSEvent e in game.Events ) {
                bool hit = e.Type == CSEventType.Hit || e.Type == CSEventType.Death;
                //if ( hit && e.PlayerId == agentId ) HitsTaken++;
                //if ( hit && e.PlayerId == enemyId ) HitsDealt++;
                //if ( e.Type == CSEventType.Shot && e.PlayerId == agentId ) ShotsFired++;

                if ( e.Type == CSEventType.Hit ) {
                    reward += ( e.PlayerId == agentId ) ? -0.1f : 0.1f;
                }
                else if ( e.Type == CSEventType.Death && e.PlayerId == enemyId ) {
                    reward += 0.1f;
                }
            }

            return reward;
        }

        // ------------------------------------------------------------------ State
        public CSGameState  GetState() { return CSGameState.Create( game, agentId, enemyId ); }
        public bool         IsTerminal() { return game.Result != CSGameResult.PLAYING; }

        // ------------------------------------------------------------------ Logging
        private readonly CSTrainingLog log = new CSTrainingLog();
        public void EndEpisode( double episodeReward ) { }
        public void PrintLog( int episode, int total, double seconds ) { log.Print( episode, total, seconds ); log.Reset(); }
    }
}
