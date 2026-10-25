namespace arcade {
    public class FirstVisitMonteCarlo {
        record TimeStep {
            public int t;
            public int state;
            public int action;
            public float reward;
        }

        private MonteCarloInterface mc;
        public FirstVisitMonteCarlo( MonteCarloInterface mc ) {
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

                    bool visited = false;
                    for ( int i = 0; i <= t - 1; i++ ) {
                        if ( ep.state == episode[i].state && ep.action == episode[i].action ) {
                            visited = true;
                            break;
                        }
                    }

                    if ( visited ) {
                        continue;
                    }

                    N[ep.state]++;
                    Q[ep.state, ep.action] = Q[ep.state, ep.action] + ( G - Q[ep.state, ep.action] ) / N[ep.state];
                    Policy[ep.state] = ArgMaxActions( Q, ep.state, Gamma );
                }
            }

            return Policy;
        }

        public int[] PolicyControl_EpsilonSoft( int EpisodeCount, double Epsilon, double Gamma = 1.0 ) {
            int[]       Policy = new int[mc.NumStates];
            int[]       N = new int[mc.NumStates];
            double[,]   Q = new double[mc.NumStates, mc.NumActions];

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                List<TimeStep> episode = GenerateEpisode(Policy, false);
                double G = 0;
                for ( int t = episode.Count - 1; t >= 0; t-- ) {
                    TimeStep ep = episode[t];
                    G = Gamma * G + ep.reward;

                    bool visited = false;
                    for ( int i = 0; i <= t - 1; i++ ) {
                        if ( ep.state == episode[i].state && ep.action == episode[i].action ) {
                            visited = true;
                            break;
                        }
                    }

                    if ( visited ) {
                        continue;
                    }

                    N[ep.state]++;
                    Q[ep.state, ep.action] = Q[ep.state, ep.action] + ( G - Q[ep.state, ep.action] ) / N[ep.state];
                    int maxAction = ArgMaxActions( Q, ep.state, Gamma );

                }
            }

            return Policy;
        }

        private int ArgMaxActions( double[,] Q, int s, double gamma ) {
            int maxAction = 0;
            double maxValue = double.MinValue;
            int[] actions = mc.LegalActions(s);
            foreach ( int action in actions ) {
                if ( Q[s, action] > maxValue ) {
                    maxValue = Q[s, action];
                    maxAction = action;
                }
            }
            return maxAction;
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
    }
}
