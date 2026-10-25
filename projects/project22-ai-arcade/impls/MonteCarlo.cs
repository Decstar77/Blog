namespace arcade {
    public class MonteCarloFirstVisitOnPolicy {
        record TimeStep {
            public int t;
            public int state;
            public int action;
            public float reward;
        }

        private MonteCarloInterface mc;
        public MonteCarloFirstVisitOnPolicy( MonteCarloInterface mc ) {
            this.mc = mc;
        }

        public double[] PolicyEvaluation( int[] Policy, int EpisodeCount, double Gamma = 1.0 ) {
            double[] V = new double[mc.NumStates];
            int[]    N = new int[mc.NumStates];
            double[] R = new double[mc.NumStates];

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
                    R[episode[t].state] = R[episode[t].state] + ( G - R[episode[t].state] ) / N[episode[t].state];
                }
            }
            return V;
        }

        public int[] PolicyControl_ExploringStarts( int EpisodeCount, double Gamma = 1.0 ) { // Policy Interation
            int[]       Policy = new int[mc.NumStates];
            int[]       N = new int[mc.NumStates];
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

                    N[ep.state]++;
                    Q[ep.state, ep.action] = Q[ep.state, ep.action] + ( G - Q[ep.state, ep.action] ) / N[ep.state];

                    int [] actions = mc.LegalActions(ep.state);
                    Policy[ep.state] = Helpers.ArgMax( actions.Length, ( int a ) => Q[ep.state, a] );
                }
            }

            return Policy;
        }

        public int[] PolicyControl_EpsilonSoft( int EpisodeCount, double Epsilon, double Gamma = 1.0 ) {
            double[,]   Policy = new double[mc.NumStates, mc.NumActions];
            int[]       N = new int[mc.NumStates];
            double[,]   Q = new double[mc.NumStates, mc.NumActions];

            for ( int s = 0; s < mc.NumStates; s++ )
                for ( int a = 0; a < mc.NumActions; a++ )
                    Policy[s, a] = 1.0 / mc.NumActions;

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                List<TimeStep> episode = GenerateEpisode(Policy);
                double G = 0;
                for ( int t = episode.Count - 1; t >= 0; t-- ) {
                    TimeStep ep = episode[t];
                    G = Gamma * G + ep.reward;

                    if ( HasBeenVisited( ep, t, episode ) ) {
                        continue;
                    }

                    N[ep.state]++;
                    Q[ep.state, ep.action] = Q[ep.state, ep.action] + ( G - Q[ep.state, ep.action] ) / N[ep.state];

                    int [] actions = mc.LegalActions(ep.state);
                    int maxAction  = Helpers.ArgMax( actions.Length, ( int a ) => Q[ep.state, a] );
                    
                    for ( int a = 0; a < mc.NumActions; a++ ) {
                        if ( a == maxAction ) {
                            Policy[ep.state, a] = 1 - Epsilon + (Epsilon / mc.NumActions);
                        } else {
                            Policy[ep.state, a] = Epsilon / mc.NumActions;
                        }
                    }
                }
            }

            // Turn it into a deterministic policy
            int[] DPolicy = new int[mc.NumStates];
            for ( int s = 0; s < mc.NumStates; s++ )
                for ( int a = 0; a < mc.NumActions; a++ ) {
                    int [] actions = mc.LegalActions(s);
                    DPolicy[s] = Helpers.ArgMax( actions.Length, ( int a ) => Q[s, a] );
                }
            

            return DPolicy;
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
            while ( mc.IsTerminal() == false ) {
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
            int[] startingActions = mc.LegalActions(mc.State());
            List<TimeStep> episode = new List<TimeStep>();
            while ( mc.IsTerminal() == false ) {
                int t = episode.Count;
                int s = mc.State();
                int a = Helpers.Choose(mc.NumActions, (int i) => Policy[s, i]);
                float r = mc.Step( a );
                episode.Add( new TimeStep() { t = t, state = s, action = a, reward = r } );
            }
            return episode;
        }
    }

    public class MonteCarloEveryVistOffPolicy {
        record TimeStep {
            public int t;
            public int state;
            public int action;
            public float reward;
        }

        private MonteCarloInterface mc;
        public MonteCarloEveryVistOffPolicy( MonteCarloInterface mc ) {
            this.mc = mc;
        }

        public double[,] PolicyEvaluation( int[,] Policy, int EpisodeCount, double Gamma = 1.0 ) {
            double[,]   C = new double[mc.NumStates, mc.NumActions];
            double[,]   Q = new double[mc.NumStates, mc.NumActions];
            double[,]   B = new double[mc.NumStates, mc.NumActions];

            for ( int s = 0; s < B.Length; s++ ) {
                for ( int a = 0; a < mc.NumActions; a++ ) {
                    B[s, a] = 1.0 / mc.NumActions;
                }
            }

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                List<TimeStep> episode = GenerateEpisode(B);

                double G = 0;
                double W = 1;
                for ( int t = episode.Count; t>=0; t-- ) {
                    int s = episode[t].state;
                    int a = episode[t].action;

                    G = Gamma * G + episode[t].reward;
                    C[s, a] = C[s, a] + W;
                    Q[s, a] = Q[s, a] + W / C[s, a] * ( G - Q[s, a] );
                    W = W * ((double)Policy[s, a] / (double)B[s, a]);

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
            double[,]   B = new double[mc.NumStates, mc.NumActions];

            for ( int s = 0; s < B.Length; s++ ) {
                for ( int a = 0; a < mc.NumActions; a++ ) {
                    B[s, a] = 1.0 / mc.NumActions;
                }
            }

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                List<TimeStep> episode = GenerateEpisode(B);

                double G = 0;
                double W = 1;
                for ( int t = episode.Count; t >= 0; t-- ) {
                    int s = episode[t].state;
                    int a = episode[t].action;

                    G = Gamma * G + episode[t].reward;
                    C[s, a] = C[s, a] + W;
                    Q[s, a] = Q[s, a] + W / C[s, a] * ( G - Q[s, a] );

                    Policy[s] = Helpers.ArgMax( mc.NumActions, ( int i ) => Q[s, i] );
                    if ( a != Policy[s] ) {
                        break;
                    }

                    W = W / B[s, a];
                }
            }

            return Q;
        }

        private List<TimeStep> GenerateEpisode( double[,] Policy ) {
            mc.NewGame();
            int[] startingActions = mc.LegalActions(mc.State());
            List<TimeStep> episode = new List<TimeStep>();
            while ( mc.IsTerminal() == false ) {
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
