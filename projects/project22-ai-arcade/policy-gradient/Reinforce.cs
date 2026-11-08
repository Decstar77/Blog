using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace arcade {
    public struct CSGameState {
        public double[] X;
    }

    public class CSReinforce {
        private CSBot           bot;
        private CSLite          game;
        private CSMap           map;
        private int?            seed;
        private int             d;
        private Random          rng;

        private List<CSPlayerAction>    actions;

        public CSReinforce( CSMap map, int d, int? seed ) {
            this.map = map;
            this.seed = seed;
            this.bot = new CSBot( 1, seed );
            this.game = new CSLite( map, seed );
            this.actions = new List<CSPlayerAction>();
            this.d = d;
            if ( seed != null ) {
                this.rng = new Random( seed.Value );
            }
            else {
                this.rng = new Random();
            }
        }

        public CSGameState GetState() {
            CSGameState s = new CSGameState();
            CSPlayer player = game.Players[0];
            s.X = new double[] {
                1, player.Health / 100.0, player.FireCooldown / CSSettings.GunFireCooldown, 
                Math.Cos(player.Orientation) / Math.Sin(player.Orientation), player.Position.X / game.WorldWidth, player.Position.Y / game.WorldHeight

            };
            return s;
        }

        public float Step( CSPlayerAction policyAction ) {
            CSPlayerAction botAction = bot.Think( game, (float)CSSettings.InvGameTickRate );
            actions.Clear();
            actions.Add( botAction );
            actions.Add( policyAction );
            game.Step( actions );

            float reward = 0;
            if ( game.Result == CSGameResult.WIN ) {
                reward = 1;
            }
            else if ( game.Result == CSGameResult.LOSS ) {
                reward = -1;
            }

            foreach ( CSEvent e in game.Events ) {
                if ( e.Type == CSEventType.Hit ) {
                    reward += ( e.PlayerId == 0 ) ? -0.1f : 0.1f;
                }
            }

            return reward;
        }
        public double[] InitialTheta() {
            double[] theta = new double[d];
            for ( int i = 0; i < d; i++ ) {
                theta[i] = rng.NextDouble() * 2 - 1;
            }
            return theta;
        }

        public CSPlayerAction Think( CSGameState s, double[] theta ) {
            for ( int i = 0; i < s.X.Length; i++ ) {
                double x = s.X[i];
               
            }
        }

        public double[] GradLogProb( CSPlayerAction action, CSGameState s, double[] theta ) {
            throw new NotImplementedException();
        }

        public bool IsTerminal() { return game.Result != CSGameResult.PLAYING; }
        public void NewGame() { game = new CSLite( map, seed ); }
    }

    public class Reinforce {
        struct StateActionReward {
            public CSGameState      s;
            public CSPlayerAction   a;
            public float            r;
        }

        public double[] PolicyControl( CSReinforce rl, int EpisodeCount, double Alpha, int? seed = null ) {
            List<StateActionReward> episode = new List<StateActionReward>();
            List<double> returns = new List<double>();

            double[] theta = rl.InitialTheta();
            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                episode.Clear();
                rl.NewGame();
                do {
                    CSGameState    s = rl.GetState();
                    CSPlayerAction a = rl.Think( s, theta );
                    float          r = rl.Step( a );
                    episode.Add( new StateActionReward() { s = s, a = a, r = r } );
                } while ( rl.IsTerminal() == false );

                returns.Clear();
                double G = 0;
                for ( int t = episode.Count - 1; t >= 0; t-- ) {
                    G = episode[t].r + G;
                    returns.Add( G );
                }
                returns.Reverse();

                // Baseline: normalise returns.
                double mean = returns.Average();
                double std  = Math.Sqrt( returns.Average( g => (g - mean) * (g - mean) ) ) + 1e-8;

                double[] update = new double[theta.Length];
                for ( int t = 0; t < episode.Count; t++ ) {
                    double adv = ( returns[t] - mean ) / std;
                    double[] grad = rl.GradLogProb( episode[t].a, episode[t].s, theta );
                    for ( int i = 0; i < theta.Length; i++ ) update[i] += adv * grad[i];
                }
                for ( int i = 0; i < theta.Length; i++ ) theta[i] += Alpha * update[i];
            }

            return theta;
        }
    }
}
