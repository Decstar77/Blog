
namespace arcade {
    public class DynamicProgramming {
        private ArcadeGame game;
        private TabularMDP mdp;

        public DynamicProgramming( ArcadeGame game, TabularMDP mdp ) {
            this.game = game;
            this.mdp = mdp;
        }

        public double[] PolicyEvaluation( int[] policy ) {
            double [] V = new double[mdp.NumStates];
            double delta = 0;
            while ( true ) {
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
                            sum += transition.Prob * ( transition.Reward + V[transition.NextState] );
                        }
                    }
                    V[s] = sum;
                    delta = Math.Max( delta, Math.Abs( v - V[s] ) );
                }

                if ( delta < 0.001 ) {
                    break;
                }
            }

            return V;
        }

        public int[] PolicyIteration() {
            int [] Policy = new int[mdp.NumStates];
            while ( true ) {
                double[] V = PolicyEvaluation( Policy );

                bool policyStable = true;
                for ( int s = 0; s < mdp.NumStates; s++ ) {
                    int oldAction = Policy[s];
                    Policy[s] = ArgMaxActions( V, s );
                    if ( oldAction != Policy[s] ) {
                        policyStable = false;
                    }
                }

                if ( policyStable ) {
                    break;
                }
            }

            return Policy;
        }

        private int ArgMaxActions( double[]V, int s ) {
            int maxAction = 0;
            double maxValue = double.MinValue;
            int[] actions = mdp.LegalActions(s);
            foreach ( int action in actions ) {
                double sum = 0;
                MdpTransition[] transitions = mdp.Transitions( s, action );
                foreach ( MdpTransition transition in transitions ) {
                    sum += transition.Reward + V[transition.NextState];
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

