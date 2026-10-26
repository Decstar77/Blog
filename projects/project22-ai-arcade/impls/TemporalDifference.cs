namespace arcade {
    public class TemporalDifference {

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
            return V;
        }
    }
}
