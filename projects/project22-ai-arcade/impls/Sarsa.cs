namespace arcade {
    public class Sarsa {
        private ReinforcementLearning rl;

        public Sarsa( ReinforcementLearning rl ) {
            this.rl = rl;
        }

        public void PolicyControl_OneStep( int EpisodeCount, double Alpha, double Epsilon ) {
            double[,] Q = new double[rl.NumStates, rl.NumActions];
            for ( int s = 0; s < rl.NumStates; s++ )
                for ( int a = 0; a < rl.NumActions; a++ )
                    Q[s, a] = 1.0 / rl.NumActions;
                

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                rl.NewGame();
                int state = rl.State();
                int action = Helpers.EpsilonGreedy( Epsilon, rl.NumActions, ( int a ) => Q[state, a] );
                while ( rl.IsTerminal() == false ) {
                    float reward = rl.Step( action );
                    int nextState = rl.State();
                    int nextAction = Helpers.EpsilonGreedy( Epsilon, rl.NumActions, ( int a ) => Q[nextState, a] );
                    Q[state, action] += Alpha * ( reward + Q[nextState, nextAction] - Q[state, action] );
                    state = nextState;
                    action = nextAction;
                }
            }
        }

        public void PolicyControl_NStep( int N ) {

        }
    }
}
