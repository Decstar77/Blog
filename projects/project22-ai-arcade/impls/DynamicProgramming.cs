
namespace arcade {
    public class DynamicProgramming {
        private DPInterface mdp;

        public DynamicProgramming( DPInterface mdp ) {
            this.mdp = mdp;
        }

        public double[] PolicyEvaluation( int[] policy, double gamma = 1.0, double theta = 0.001 ) {
            double [] V = new double[mdp.NumStates];
            while ( true ) {
                double delta = 0;
                for ( int s = 0; s < mdp.NumStates; s++ ) {
                    double v = V[s];
                    int[] actions = mdp.LegalActions( s );
                    double sum = 0;

                    // This loop can be collpased but it's understanding is more important and it's clearer this way.
                    foreach ( int a in actions ) {
                        if ( a != policy[s] ) {
                            continue;
                        }

                        MdpTransition[] transitions = mdp.Transitions( s, a );
                        foreach ( MdpTransition transition in transitions ) {
                            sum += transition.Prob * ( transition.Reward + gamma * V[transition.NextState] );
                        }
                    }
                    V[s] = sum;
                    delta = Math.Max( delta, Math.Abs( v - V[s] ) );
                }

                if ( delta < theta ) {
                    break;
                }
            }

            return V;
        }

        public int[] PolicyIteration( double gamma = 1.0 ) {
            int [] Policy = new int[mdp.NumStates];
            for ( int s = 0; s < mdp.NumStates; s++ ) {
                if ( mdp.IsTerminal( s ) ) {
                    Policy[s] = 0;
                    continue;
                }
                Policy[s] = mdp.LegalActions( s )[0];
            }

            while ( true ) {
                double[] V = PolicyEvaluation( Policy, gamma );

                bool policyStable = true;
                for ( int s = 0; s < mdp.NumStates; s++ ) {
                    int oldAction = Policy[s];
                    int newAction = ArgMaxActions( V, s, gamma );
                    if ( QValue( V, s, newAction, gamma ) > QValue( V, s, oldAction, gamma ) + 1e-9 ) {
                        Policy[s] = newAction;
                        policyStable = false;
                    }
                }

                if ( policyStable ) {
                    break;
                }
            }

            return Policy;
        }

        private double QValue( double[] V, int s, int a, double gamma ) {
            double Q = 0;
            MdpTransition[] transitions = mdp.Transitions( s, a );
            foreach ( MdpTransition transition in transitions ) {
                Q += transition.Prob * ( transition.Reward + gamma * V[transition.NextState] );
            }
            return Q;
        }

        private int ArgMaxActions( double[] V, int s, double gamma ) {
            int maxAction = 0;
            double maxValue = double.MinValue;
            int[] actions = mdp.LegalActions(s);
            foreach ( int action in actions ) {
                double sum = 0;
                MdpTransition[] transitions = mdp.Transitions( s, action );
                foreach ( MdpTransition transition in transitions ) {
                    sum += transition.Prob * ( transition.Reward + gamma * V[transition.NextState] );
                }
                if ( sum > maxValue ) {
                    maxValue = sum;
                    maxAction = action;
                }
            }
            return maxAction;
        }
    };
}

