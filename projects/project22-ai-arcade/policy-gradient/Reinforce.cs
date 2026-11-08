namespace arcade {
    public interface PolicyGradientInterface<TState, TAction, TCache> {
        public double[]             InitialTheta();
        public void                 NewGame();
        public TState               GetState();
        public (TAction, TCache)    Think( TState s, double[] theta );   // sample a ~ pi(.|s, theta)
        public float                Step( TAction a );                   // apply a, return the reward
        public bool                 IsTerminal();
        public void                 AccumulateGradLogProb( TState s, TAction a, TCache cache, double[] theta, double scale, double[] into ); // into += scale * grad_theta ln pi(a|s, theta)

        // Logging hooks.
        public void                 EndEpisode( double episodeReward );
        public void                 PrintLog( int episode, int total, double seconds );
    }

    public class Reinforce {
        struct StateActionReward<TState, TAction, TCache> {
            public TState   s;
            public TAction  a;
            public TCache   c;
            public float    r;
        }

        // Prints a summary line every LogEvery episodes (0 = silent).
        public double[] PolicyControl<TState, TAction, TCache>( PolicyGradientInterface<TState, TAction, TCache> rl, int EpisodeCount, double Alpha, double Gamma, int? seed = null, int LogEvery = 10 ) {
            var clock = System.Diagnostics.Stopwatch.StartNew();

            var episode = new List<StateActionReward<TState, TAction, TCache>>();
            List<double> returns = new List<double>();
            double[] theta = rl.InitialTheta();
            double[] update = new double[theta.Length];

            double runningMean = 0;
            double runningVar = 1;

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                episode.Clear();
                rl.NewGame();
                do {
                    TState              s = rl.GetState();
                    (TAction a, TCache c) = rl.Think( s, theta );
                    float               r = rl.Step( a );
                    episode.Add( new StateActionReward<TState, TAction, TCache>() { s = s, a = a, c = c, r = r } );
                } while ( rl.IsTerminal() == false );

                // Logging ================
                double episodeReward = 0;
                foreach ( var step in episode ) episodeReward += step.r;
                rl.EndEpisode( episodeReward );
                if ( LogEvery > 0 && ( episodeIndex + 1 ) % LogEvery == 0 ) {
                    rl.PrintLog( episodeIndex + 1, EpisodeCount, clock.Elapsed.TotalSeconds );
                }

                // Returns ================
                returns.Clear();
                double G = 0;
                for ( int t = episode.Count - 1; t >= 0; t-- ) {
                    G = episode[t].r + Gamma * G;
                    returns.Add( G );
                }
                returns.Reverse();

                // Baseline ================
                double currentMean = returns.Average();
                double currentVar  = returns.Average( g => (g - runningMean) * (g - runningMean) );
                double std         = Math.Sqrt( Math.Max( runningVar, 1e-2 ) );

                Array.Clear( update );
                for ( int t = 0; t < episode.Count; t++ ) {
                    double adv = ( returns[t] - runningMean ) / ( std );
                    rl.AccumulateGradLogProb( episode[t].s, episode[t].a, episode[t].c, theta, adv, update );
                }
                for ( int i = 0; i < theta.Length; i++ ) {
                    theta[i] += Alpha * update[i] / episode.Count;
                }

                const double Beta = 0.05;
                runningMean += Beta * ( currentMean - runningMean );
                runningVar += Beta * ( currentVar - runningVar );
            }

            return theta;
        }
    }
}
