using System;
using System.Collections.Generic;
using System.Text;

namespace arcade {
    public class CSGameState { }

    public interface ReinforcePolicy {
        public CSPlayerAction Step( CSGameState s, float[] theta );
    }

    public interface ReinforceInterface {
        public void         NewGame();
        public bool         IsTerminal();
        public CSGameState  GetState();
        public float        Step(CSPlayerAction action);
    }

    public class Reinforce {
        struct StateActionReward {
            public CSGameState      s;
            public CSPlayerAction   a;
            public float            r;
        }

        private ReinforceInterface rl;
        public Reinforce( ReinforceInterface rl ) {
            this.rl = rl;
        }

        public void PolicyControl( ReinforcePolicy Policy, int EpisodeCount, double Alpha, double DiscountRate, int d ) {
            float[] theta = new float[d];
            List<StateActionReward> episode = new List<StateActionReward>();
            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                episode.Clear();
                rl.NewGame();
                do {
                    CSGameState    s = rl.GetState();
                    CSPlayerAction a = Policy.Step( s, theta );
                    float          r = rl.Step( a );
                    episode.Add( new StateActionReward() { s = s, a = a, r = r } );
                } while ( rl.IsTerminal() == false );

                for ( int t = 0; t < episode.Count; t++ ) {
                    double G = 0;
                    for ( int k = t + 1; k < episode.Count; k++ ) {
                        G += Math.Pow( DiscountRate, k - t - 1 ) * episode[k].r;
                    }

                }
            }
        }
    }
}
