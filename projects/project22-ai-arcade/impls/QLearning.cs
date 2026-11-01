namespace arcade {
    public class QLearning {
        private TabularModelFreeInterface rl;
        private record StateActionReward {
            public int s;
            public int a;
            public float r;
        }

        public QLearning( TabularModelFreeInterface rl ) {
            this.rl = rl;
        }

        public int[] PolicyControl_OneStep( int EpisodeCount, double Alpha, double Epsilon, double Gamma = 1.0 ) {
            double[,] Q = InitialQ();

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                rl.NewGame();
                while ( rl.IsTerminal() == false ) {
                    int state = rl.State();
                    int action = EpsilonGreedyAction( Q, state, Epsilon );
                    float reward = rl.Step( action );
                    int nextState = rl.State();
                    Q[state, action] += Alpha * ( reward + Gamma * MaxQ( Q, nextState ) - Q[state, action] );
                }
            }

            return GreedyPolicy( Q );
        }

        public int[] PolicyControl_DoubleQ( int EpisodeCount, double Alpha, double Epsilon, double Gamma = 1.0 ) {
            double[,] Q1 = InitialQ();
            double[,] Q2 = InitialQ();
            double[,] QMean = InitialQ(); // Behaviour and final policy use both tables

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                rl.NewGame();
                while ( rl.IsTerminal() == false ) {
                    int state = rl.State();
                    int action = EpsilonGreedyAction( QMean, state, Epsilon );
                    float reward = rl.Step( action );
                    int nextState = rl.State();

                    bool updateFirst = Random.Shared.Next( 2 ) == 0;
                    double[,] QSelect = updateFirst ? Q1 : Q2;
                    double[,] QValue = updateFirst ? Q2 : Q1;
                    QSelect[state, action] += Alpha * ( reward + Gamma * CrossQ( QSelect, QValue, nextState ) - QSelect[state, action] );
                    QMean[state, action] = ( Q1[state, action] + Q2[state, action] ) / 2;
                }
            }

            return GreedyPolicy( QMean );
        }

        // Rewards are only summed along greedy actions, the target policy never takes the others.
        // The return is cut at the first exploratory action, which is tree backup with a greedy target.
        public int[] PolicyControl_NStep( int EpisodeCount, int N, double Alpha, double Epsilon, double Gamma = 1.0 ) {
            double[,] Q = InitialQ();

            List<StateActionReward> store = new List<StateActionReward>();
            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                rl.NewGame();
                store.Clear();
                int startState = rl.State();
                store.Add( new StateActionReward() { s = startState, a = EpsilonGreedyAction( Q, startState, Epsilon ), r = 0 } );
                int T = int.MaxValue;
                for ( int t = 0, tau = t - N + 1; tau < T - 1; t++ ) {
                    if ( t < T ) {
                        float r = rl.Step( store[t].a );
                        int nextState = rl.State();
                        store.Add( new StateActionReward() { s = nextState, a = EpsilonGreedyAction( Q, nextState, Epsilon ), r = r } );
                        if ( rl.IsTerminal() ) {
                            T = t + 1;
                        }
                    }
                    tau = t - N + 1;
                    if ( tau >= 0 ) {
                        double G = 0;
                        double discount = 1;
                        for ( int i = tau + 1; ; i++ ) {
                            StateActionReward step = store[i];
                            G += discount * step.r;
                            discount *= Gamma;
                            double maxQ = MaxQ( Q, step.s );
                            if ( i == tau + N || i == T || Q[step.s, step.a] != maxQ ) {
                                G += discount * maxQ;
                                break;
                            }
                        }
                        StateActionReward st = store[tau];
                        Q[st.s, st.a] += Alpha * ( G - Q[st.s, st.a] );
                    }
                }
            }

            return GreedyPolicy( Q );
        }

        private double[,] InitialQ() {
            double[,] Q = new double[rl.NumStates, rl.NumActions];
            for ( int s = 0; s < rl.NumStates; s++ ) {
                int[] actions = rl.LegalActions(s);
                foreach ( int a in actions )
                    Q[s, a] = 1.0 / rl.NumActions;
            }
            return Q;
        }

        private int EpsilonGreedyAction( double[,] Q, int s, double Epsilon ) {
            int[] actions = rl.LegalActions(s);
            if ( actions.Length == 0 ) {
                return 0;
            }
            return actions[Helpers.EpsilonGreedy( Epsilon, actions.Length, ( int i ) => Q[s, actions[i]] )];
        }

        private double MaxQ( double[,] Q, int s ) {
            int[] actions = rl.LegalActions(s);
            if ( actions.Length == 0 ) {
                return 0;
            }
            double max = double.MinValue;
            foreach ( int a in actions ) {
                max = Math.Max( max, Q[s, a] );
            }
            return max;
        }

        private double CrossQ( double[,] QSelect, double[,] QValue, int s ) {
            int[] actions = rl.LegalActions(s);
            if ( actions.Length == 0 ) {
                return 0;
            }
            return QValue[s, actions[Helpers.ArgMax( actions.Length, ( int i ) => QSelect[s, actions[i]] )]];
        }

        private int[] GreedyPolicy( double[,] Q ) {
            int[] Policy = new int[rl.NumStates];
            for ( int s = 0; s < rl.NumStates; s++ ) {
                int[] actions = rl.LegalActions(s);
                if ( actions.Length > 0 ) {
                    Policy[s] = actions[Helpers.ArgMax( actions.Length, ( int i ) => Q[s, actions[i]] )];
                }
            }
            return Policy;
        }
    }
}

