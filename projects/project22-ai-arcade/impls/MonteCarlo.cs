namespace arcade {
    public class MonteCarloFirstVisitOnPolicy {
        record TimeStep {
            public int t;
            public int state;
            public int action;
            public float reward;
        }

        private MonteCarloInterface mc;
        public int MaxEpisodeSteps = 1000;

        public MonteCarloFirstVisitOnPolicy( MonteCarloInterface mc ) {
            this.mc = mc;
        }

        public double[] PolicyEvaluation( int[] Policy, int EpisodeCount, double Gamma = 1.0 ) {
            double[] V = new double[mc.NumStates];
            int[]    N = new int[mc.NumStates];

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                List<TimeStep> episode = GenerateEpisode(Policy, false);
                double G = 0;
                for ( int t = episode.Count - 1; t >= 0; t-- ) {
                    G = Gamma * G + episode[t].reward;

                    bool visited = false;
                    for ( int i = 0; i <= t - 1; i++ ) {
                        if ( episode[t].state == episode[i].state ) {
                            visited = true;
                            break;
                        }
                    }

                    if ( visited ) {
                        continue;
                    }

                    N[episode[t].state]++;
                    V[episode[t].state] = V[episode[t].state] + ( G - V[episode[t].state] ) / N[episode[t].state];
                }
            }
            return V;
        }

        public int[] PolicyControl_ExploringStarts( int EpisodeCount, double Gamma = 1.0 ) { // Policy Interation
            int[]       Policy = new int[mc.NumStates];
            int[,]      N = new int[mc.NumStates, mc.NumActions];
            double[,]   Q = new double[mc.NumStates, mc.NumActions];

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                List<TimeStep> episode = GenerateEpisode(Policy, true);
                double G = 0;
                for ( int t = episode.Count - 1; t >= 0; t-- ) {
                    TimeStep ep = episode[t];
                    G = Gamma * G + ep.reward;

                    if ( HasBeenVisited( ep, t, episode ) ) {
                        continue;
                    }

                    N[ep.state, ep.action]++;
                    Q[ep.state, ep.action] = Q[ep.state, ep.action] + ( G - Q[ep.state, ep.action] ) / N[ep.state, ep.action];

                    Policy[ep.state] = GreedyAction( Q, ep.state );
                }
            }

            return Policy;
        }

        public int[] PolicyControl_EpsilonSoft( int EpisodeCount, double Epsilon, double Gamma = 1.0 ) {
            double[,]   Policy = new double[mc.NumStates, mc.NumActions];
            int[,]      N = new int[mc.NumStates, mc.NumActions];
            double[,]   Q = new double[mc.NumStates, mc.NumActions];

            for ( int s = 0; s < mc.NumStates; s++ ) {
                int[] actions = mc.LegalActions(s);
                foreach ( int a in actions )
                    Policy[s, a] = 1.0 / actions.Length;
            }

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                List<TimeStep> episode = GenerateEpisode(Policy);
                double G = 0;
                for ( int t = episode.Count - 1; t >= 0; t-- ) {
                    TimeStep ep = episode[t];
                    G = Gamma * G + ep.reward;

                    if ( HasBeenVisited( ep, t, episode ) ) {
                        continue;
                    }

                    N[ep.state, ep.action]++;
                    Q[ep.state, ep.action] = Q[ep.state, ep.action] + ( G - Q[ep.state, ep.action] ) / N[ep.state, ep.action];

                    // Epsilon is spread over legal actions only, illegal actions keep zero probability
                    int[] actions   = mc.LegalActions(ep.state);
                    int maxAction   = GreedyAction( Q, ep.state );
                    foreach ( int a in actions ) {
                        if ( a == maxAction ) {
                            Policy[ep.state, a] = 1 - Epsilon + (Epsilon / actions.Length);
                        } else {
                            Policy[ep.state, a] = Epsilon / actions.Length;
                        }
                    }
                }
            }

            // Turn it into a deterministic policy
            int[] DPolicy = new int[mc.NumStates];
            for ( int s = 0; s < mc.NumStates; s++ )
                DPolicy[s] = GreedyAction( Q, s );

            return DPolicy;
        }

        // Returns the legal action with the highest Q value, ties broken randomly. Terminal states return 0.
        private int GreedyAction( double[,] Q, int s ) {
            int[] actions = mc.LegalActions(s);
            if ( actions.Length == 0 ) {
                return 0;
            }
            return actions[Helpers.ArgMax( actions.Length, ( int i ) => Q[s, actions[i]] )];
        }

        private bool HasBeenVisited( TimeStep ep, int t, List<TimeStep> episode ) {
            for ( int i = 0; i <= t - 1; i++ ) {
                if ( ep.state == episode[i].state && ep.action == episode[i].action ) {
                    return true;
                }
            }
            return false;
        }

        private List<TimeStep> GenerateEpisode(int[] Policy, bool randomStart) {
            if ( randomStart ) {
                mc.RandomGame();
            }else {
                mc.NewGame();
            }

            int[] startingActions = mc.LegalActions(mc.State());
            List<TimeStep> episode = new List<TimeStep>();
            while ( mc.IsTerminal() == false && episode.Count < MaxEpisodeSteps ) {
                int t = episode.Count;
                int s = mc.State();
                int a = (t == 0 && randomStart) ? startingActions[Random.Shared.Next(startingActions.Length)] : Policy[s];
                float r = mc.Step( a );
                episode.Add( new TimeStep() { t = t, state = s, action = a, reward = r } );
            }
            return episode;
        }

        private List<TimeStep> GenerateEpisode( double[,] Policy ) {
            mc.NewGame();
            List<TimeStep> episode = new List<TimeStep>();
            while ( mc.IsTerminal() == false && episode.Count < MaxEpisodeSteps ) {
                int t = episode.Count;
                int s = mc.State();
                int a = Helpers.Choose(mc.NumActions, (int i) => Policy[s, i]);
                float r = mc.Step( a );
                episode.Add( new TimeStep() { t = t, state = s, action = a, reward = r } );
            }
            return episode;
        }
    }

    public class MonteCarloEveryVisitOffPolicy {
        record TimeStep {
            public int t;
            public int state;
            public int action;
            public float reward;
        }

        private MonteCarloInterface mc;
        public int MaxEpisodeSteps = 1000;

        public MonteCarloEveryVisitOffPolicy( MonteCarloInterface mc ) {
            this.mc = mc;
        }

        public double[,] PolicyEvaluation( double[,] Policy, int EpisodeCount, double Gamma = 1.0 ) {
            double[,]   C = new double[mc.NumStates, mc.NumActions];
            double[,]   Q = new double[mc.NumStates, mc.NumActions];
            double[,]   B = UniformBehaviourPolicy();

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                List<TimeStep> episode = GenerateEpisode(B);

                double G = 0;
                double W = 1;
                for ( int t = episode.Count - 1; t >= 0; t-- ) {
                    int s = episode[t].state;
                    int a = episode[t].action;

                    G = Gamma * G + episode[t].reward;
                    C[s, a] = C[s, a] + W;
                    Q[s, a] = Q[s, a] + W / C[s, a] * ( G - Q[s, a] );
                    W = W * ( Policy[s, a] / B[s, a] );

                    if ( W == 0 ) {
                        break;
                    }
                }
            }

            return Q;
        }

        public double[,] PolicyControl( int[] Policy, int EpisodeCount, double Gamma = 1.0 ) {
            double[,]   C = new double[mc.NumStates, mc.NumActions];
            double[,]   Q = new double[mc.NumStates, mc.NumActions];
            double[,]   B = UniformBehaviourPolicy();

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                List<TimeStep> episode = GenerateEpisode(B);

                double G = 0;
                double W = 1;
                for ( int t = episode.Count - 1; t >= 0; t-- ) {
                    int s = episode[t].state;
                    int a = episode[t].action;

                    G = Gamma * G + episode[t].reward;
                    C[s, a] = C[s, a] + W;
                    Q[s, a] = Q[s, a] + W / C[s, a] * ( G - Q[s, a] );

                    int[] actions = mc.LegalActions(s);
                    Policy[s] = actions[Helpers.ArgMax( actions.Length, ( int i ) => Q[s, actions[i]] )];
                    if ( a != Policy[s] ) {
                        break;
                    }

                    W = W / B[s, a];
                }
            }

            return Q;
        }

        // Uniform over legal actions, illegal actions get zero probability
        private double[,] UniformBehaviourPolicy() {
            double[,] B = new double[mc.NumStates, mc.NumActions];
            for ( int s = 0; s < mc.NumStates; s++ ) {
                int[] actions = mc.LegalActions(s);
                foreach ( int a in actions ) {
                    B[s, a] = 1.0 / actions.Length;
                }
            }
            return B;
        }

        private List<TimeStep> GenerateEpisode( double[,] Policy ) {
            mc.NewGame();
            List<TimeStep> episode = new List<TimeStep>();
            while ( mc.IsTerminal() == false && episode.Count < MaxEpisodeSteps ) {
                int t = episode.Count;
                int s = mc.State();
                int a = Helpers.Choose(mc.NumActions, (int i) => Policy[s, i]);
                float r = mc.Step( a );
                episode.Add( new TimeStep() { t = t, state = s, action = a, reward = r } );
            }
            return episode;
        }

    }
}
