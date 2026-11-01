namespace arcade {
    public class Sarsa {
        private ReinforcementLearning rl;
        private record StateActionReward {
            public int s;
            public int a;
            public float r;
        }

        public Sarsa( ReinforcementLearning rl ) {
            this.rl = rl;
        }

        public int[] PolicyControl_OneStep( int EpisodeCount, double Alpha, double Epsilon, double Gamma = 1.0 ) {
            double[,] Q = InitialQ();

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                rl.NewGame();
                int state = rl.State();
                int action = EpsilonGreedyAction( Q, state, Epsilon );
                while ( rl.IsTerminal() == false ) {
                    float reward = rl.Step( action );
                    int nextState = rl.State();
                    int nextAction = EpsilonGreedyAction( Q, nextState, Epsilon );
                    Q[state, action] += Alpha * ( reward + Gamma * Q[nextState, nextAction] - Q[state, action] );
                    state = nextState;
                    action = nextAction;
                }
            }

            return GreedyPolicy( Q );
        }

        public int[] PolicyControl_ExpectedSarsa( int EpisodeCount, double Alpha, double Epsilon, double Gamma = 1.0 ) {
            double[,] Q = InitialQ();

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                rl.NewGame();
                while ( rl.IsTerminal() == false ) {
                    int state = rl.State();
                    int action = EpsilonGreedyAction( Q, state, Epsilon );
                    float reward = rl.Step( action );
                    int nextState = rl.State();
                    Q[state, action] += Alpha * ( reward + Gamma * ExpectedQ( Q, nextState, Epsilon ) - Q[state, action] );
                }
            }

            return GreedyPolicy( Q );
        }

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
                        for ( int i = tau + 1; i <= Math.Min( tau + N, T ); i++ ) {
                            G += Math.Pow( Gamma, i - tau - 1 ) * store[i].r;
                        }
                        if ( tau + N < T ) {
                            StateActionReward last = store[tau + N];
                            G = G + Math.Pow( Gamma, N ) * Q[last.s, last.a];
                        }
                        StateActionReward st = store[tau];
                        Q[st.s, st.a] += Alpha * ( G - Q[st.s, st.a] );
                    }
                }
            }

            return GreedyPolicy( Q );
        }

        // Legal actions start uniform, terminal states have no legal actions so their Q stays zero
        private double[,] InitialQ() {
            double[,] Q = new double[rl.NumStates, rl.NumActions];
            for ( int s = 0; s < rl.NumStates; s++ ) {
                int[] actions = rl.LegalActions(s);
                foreach ( int a in actions )
                    Q[s, a] = 1.0 / rl.NumActions;
            }
            return Q;
        }

        // Terminal states return 0, the action is never played and its Q is zero
        private int EpsilonGreedyAction( double[,] Q, int s, double Epsilon ) {
            int[] actions = rl.LegalActions(s);
            if ( actions.Length == 0 ) {
                return 0;
            }
            return actions[Helpers.EpsilonGreedy( Epsilon, actions.Length, ( int i ) => Q[s, actions[i]] )];
        }

        // Value of s under the epsilon greedy policy. Tied greedy actions share a value, so tie breaking has no effect.
        private double ExpectedQ( double[,] Q, int s, double Epsilon ) {
            int[] actions = rl.LegalActions(s);
            if ( actions.Length == 0 ) {
                return 0;
            }
            double sum = 0;
            double max = double.MinValue;
            foreach ( int a in actions ) {
                sum += Q[s, a];
                max = Math.Max( max, Q[s, a] );
            }
            return ( 1 - Epsilon ) * max + Epsilon * sum / actions.Length;
        }

        // Turn it into a deterministic policy
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
