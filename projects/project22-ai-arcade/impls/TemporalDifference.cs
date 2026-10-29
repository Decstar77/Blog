namespace arcade {
    public class TemporalDifference {
        private record StateReward {
            public int s;
            public float r;
        }

        private MonteCarloInterface rl;
        
        public TemporalDifference(MonteCarloInterface rl) {
            this.rl = rl;
        }

        public double[] PolicyEvaluation_OneStep( int[] Policy, int EpisodeCount, double Alpha, double Gamma = 1.0 ) {
            double[] V = new double[rl.NumStates];
            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                rl.NewGame();
                while ( rl.IsTerminal() == false ) {
                    int s = rl.State();
                    int a = Policy[s];
                    float r = rl.Step( a );
                    int sp = rl.State();
                    V[s] = V[s] + Alpha * ( r + Gamma * V[sp] - V[s] );
                }
            }
            return V;
        }

        public double[] PolicyEvaluation_NStep( int[] Policy, int EpisodeCount, int N, double Alpha, double Gamma = 1.0 ) {
            double[] V = new double[rl.NumStates];
            List<StateReward> store = new List<StateReward>();
            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                rl.NewGame();
                store.Add( new StateReward() { s = rl.State(), r = 0 } );
                int T = int.MaxValue;
                for ( int t = 0, tau = t - N + 1; t <= T - 1; t++ ) {
                    if ( t < T ) {
                        StateReward s = store[t];
                        int a = Policy[s.s];
                        float r = rl.Step( a );
                        store.Add( new StateReward() { s = rl.State(), r = r } );
                        if ( rl.IsTerminal() ) {
                            T = t + 1;
                        }
                    }
                    tau = t - N + 1;
                    if ( tau >= 0 ) {
                        double G= 0;
                        for ( int i = tau + 1; i <= Math.Min(tau + N, T); i++ ) {
                            G += Math.Pow( Gamma, i - tau - 1 ) * store[i].r;
                        }
                        if ( tau + N < T ) {
                            G = G + Math.Pow( Gamma, N ) * V[store[tau + N].s];
                        }
                        StateReward st = store[tau];
                        V[st.s] += Alpha * ( G - V[st.s] );

                    }
                }
            }
            return V;
        }
    }
}
