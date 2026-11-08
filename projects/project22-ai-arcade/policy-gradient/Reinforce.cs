using System.Numerics;

namespace arcade {
    public struct CSGameState {
        public double[] X;
    }

    public struct GradData {
        public double[] m;
        public double[] r;
        public double[] f;
        public int ma;
        public int ra;
        public int fa;
    }

    public class CSReinforce {
        private CSBot           bot;
        private CSLite          game;
        private CSMap           map;
        private int?            seed;
        private Random          rng;

        private List<CSPlayerAction>    actions;

        // Agent = the team-0 player ('1' on the map), so CSGameResult.WIN means the agent won. Players are created in map-scan order, so this index differs per map (map001: 1, map002: 0).
        private int             agentId;
        private int             enemyId;

        public CSReinforce( CSMap map, int? seed ) {
            this.map = map;
            this.seed = seed;
            this.actions = new List<CSPlayerAction>();
            if ( seed != null ) {
                this.rng = new Random( seed.Value );
            }
            else {
                this.rng = new Random();
            }
            NewGame();
        }

        // ------------------------------------------------------------------ Features

        public const int    RayCount      = 8;
        public const float  RayRange      = 300;   // 6 cells
        public const int    FeatureCount  = 7 + 7 + 2 + RayCount + 5 + 1; // = 30
        public const int    D             = FeatureCount * 14;

        public CSGameState GetState() { return GetState( game, agentId, enemyId ); }

        // Features for any game, so a trained theta can drive the agent in a game this class doesn't own (e.g. the rendered one).
        public static CSGameState GetState( CSLite game, int agentId, int enemyId ) {
            CSPlayer me    = game.Players[agentId];
            CSPlayer enemy = game.Players[enemyId];

            Vector2 fwd   = me.Forward;
            Vector2 right = new Vector2( -fwd.Y, fwd.X );
            float   diag  = MathF.Sqrt( game.WorldWidth * game.WorldWidth + game.WorldHeight * game.WorldHeight );

            double[] x = new double[FeatureCount];
            int i = 0;

            // Bias + self (7)
            x[i++] = 1;
            x[i++] = me.Health / (double)CSSettings.PlayerHealth;
            x[i++] = me.FireCooldown / CSSettings.GunFireCooldown;
            x[i++] = MathF.Cos( me.Orientation );
            x[i++] = MathF.Sin( me.Orientation );
            x[i++] = me.Position.X / game.WorldWidth;
            x[i++] = me.Position.Y / game.WorldHeight;

            // Enemy, egocentric (7)
            Vector2 toEnemy  = enemy.Position - me.Position;
            float   dist     = toEnemy.Length();
            float   bearing  = RelativeBearing( me, toEnemy );
            Vector2 toMeDir  = dist > 1e-3f ? -toEnemy / dist : Vector2.Zero;
            x[i++] = MathF.Sin( bearing );                       // which side the enemy is on: turn this way
            x[i++] = MathF.Cos( bearing );                       // 1 = dead ahead
            x[i++] = dist / diag;
            x[i++] = game.HasLineOfSight( me.Position, enemy.Position ) ? 1 : 0;
            x[i++] = enemy.Health / (double)CSSettings.PlayerHealth;
            x[i++] = Vector2.Dot( enemy.Forward, toMeDir );      // 1 = enemy is aiming straight at me
            x[i++] = enemy.FireCooldown / CSSettings.GunFireCooldown;

            // Navigation: bearing to the next BFS waypoint toward the enemy (2)
            Vector2 waypoint = CSBot.NextWaypoint( game, me.Position, enemy.Position ) ?? enemy.Position;
            float   wpBearing = RelativeBearing( me, waypoint - me.Position );
            x[i++] = MathF.Sin( wpBearing );
            x[i++] = MathF.Cos( wpBearing );

            // Wall rays relative to facing, 1 = nothing within RayRange (8)
            for ( int r = 0; r < RayCount; r++ ) {
                float   a   = me.Orientation + r * MathF.PI * 2 / RayCount;
                Vector2 end = me.Position + new Vector2( MathF.Cos( a ), MathF.Sin( a ) ) * RayRange;
                float   t   = 1;
                foreach ( var c in game.Collider ) {
                    float hit = CSLite.SegmentBoxEntry( me.Position, end, c );
                    if ( hit >= 0 && hit < t ) t = hit;
                }
                x[i++] = t;
            }

            // Nearest enemy bullet, in my local frame (5). All zeros if there is none.
            int   best = -1;
            float bestDist = float.MaxValue;
            for ( int b = 0; b < game.Bullets.Count; b++ ) {
                CSBullet bullet = game.Bullets[b];
                if ( game.Players[bullet.PlayerId].Team == me.Team ) continue;
                float d2 = Vector2.DistanceSquared( bullet.Position, me.Position );
                if ( d2 < bestDist ) { bestDist = d2; best = b; }
            }
            if ( best >= 0 ) {
                CSBullet bullet = game.Bullets[best];
                Vector2  rel    = bullet.Position - me.Position;
                float    bd     = rel.Length();
                Vector2  bDir   = new Vector2( MathF.Cos( bullet.Direction ), MathF.Sin( bullet.Direction ) );
                x[i++] = 1;                                                    // a bullet exists
                x[i++] = Vector2.Dot( rel, fwd ) / diag;                       // ahead / behind
                x[i++] = Vector2.Dot( rel, right ) / diag;                     // right / left
                x[i++] = bd / diag;
                x[i++] = bd > 1e-3f ? Vector2.Dot( bDir, -rel / bd ) : 0;      // 1 = flying straight at me
            }
            else {
                i += 5;
            }

            // Time (1)
            x[i++] = game.TickCount * CSSettings.InvGameTickRate / CSSettings.MaxRoundTime;

            System.Diagnostics.Debug.Assert( i == FeatureCount );
            return new CSGameState { X = x };
        }

        // Angle from my facing to `dir`, in [-pi, pi]. Positive = turn with positive Rotation.
        private static float RelativeBearing( CSPlayer me, Vector2 dir ) {
            if ( dir.LengthSquared() < 1e-6f ) return 0;
            return CSLite.WrapAngle( MathF.Atan2( dir.Y, dir.X ) - me.Orientation );
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
                    reward += ( e.PlayerId == agentId ) ? -0.1f : 0.1f;
                }
                else if ( e.Type == CSEventType.Death && e.PlayerId == enemyId ) {
                    reward += 0.1f;
                }
            }

            return reward;
        }
        
        public static double[] HeadProbs( double[] x, double[] theta, int startingRow, int rowCount ) {
            double[] p = new double[rowCount];
            double max = double.NegativeInfinity;
            for ( int row = 0; row < rowCount; row++ ) {
                int off = (startingRow + row) * FeatureCount;
                double g = 0;
                for ( int i = 0; i < FeatureCount; i++ ) {
                    g += x[i] * theta[off + i];
                }
                p[row] = g;
                max = Math.Max( max, g );
            }

            double sum = 0;
            for ( int r = 0; r < rowCount; r++ ) { p[r] = Math.Exp( p[r] - max ); sum += p[r]; } // subtract max: no overflow
            for ( int r = 0; r < rowCount; r++ ) p[r] /= sum;
            return p;
        }

        static readonly (bool L, bool R, bool U, bool D)[] Moves = {
            (false,false,false,false), (true,false,false,false), (false,true,false,false),
            (false,false,true,false),  (false,false,false,true), (true,false,true,false),
            (true,false,false,true),   (false,true,true,false),  (false,true,false,true) 
        };

        // Run the trained policy for the team-0 player of an external game.
        public CSPlayerAction Act( CSLite external, double[] theta ) {
            int me    = external.Players.FindIndex( p => p.Team == 0 );
            int enemy = external.Players.FindIndex( p => p.Team == 1 );
            (CSPlayerAction action, _) = Think( GetState( external, me, enemy ), theta );
            action.PlayerId = me;
            return action;
        }

        public (CSPlayerAction, GradData) Think( CSGameState s, double[] theta ) {
            GradData grad = new GradData();
            grad.m = HeadProbs(s.X, theta, 0, 9);
            grad.r = HeadProbs(s.X, theta, 9, 3);
            grad.f = HeadProbs(s.X, theta, 12, 2);
            grad.ma = Helpers.Choose( grad.m.Length, i => grad.m[i], rng );
            grad.ra = Helpers.Choose( grad.r.Length, i => grad.r[i], rng );
            grad.fa = Helpers.Choose( grad.f.Length, i => grad.f[i], rng );
            CSPlayerAction action = new CSPlayerAction();
            action.PlayerId     = agentId;
            action.MoveLeft     = Moves[grad.ma].L;
            action.MoveRight    = Moves[grad.ma].R;
            action.MoveUp       = Moves[grad.ma].U;
            action.MoveDown     = Moves[grad.ma].D;
            action.Rotation     = (float)grad.ra - 1.0f;
            action.Shoot        = grad.fa == 0;
            return (action, grad);
        }

        public void HeadGrads( double[] x, double[] p, double[] grad, int a, int startingRow, int rowCount ) {
            for ( int k = 0; k < rowCount; k++ ) {
                int off = (startingRow + k) * FeatureCount;
                double coef = ( k == a ? 1 : 0 ) - p[k];
                for ( int i = 0; i < FeatureCount; i++ ) {
                    grad[off + i] = x[i] * coef;
                }
            }
        }

        public double[] GradLogProb( CSPlayerAction action, CSGameState s, GradData g, double[] theta ) {
            double[] grad = new double[D];
            HeadGrads(s.X, g.m, grad, g.ma, 0, 9);
            HeadGrads(s.X, g.r, grad, g.ra, 9, 3);
            HeadGrads(s.X, g.f, grad, g.fa, 12, 2);
            return grad;
        }

        public void NewGame() {
            game = new CSLite( map, seed );
            agentId = game.Players.FindIndex( p => p.Team == 0 );
            enemyId = game.Players.FindIndex( p => p.Team == 1 );
            bot = new CSBot( enemyId, seed );
        }

        public bool     IsTerminal() { return game.Result != CSGameResult.PLAYING; }
        public double[] InitialTheta() { return new double[D]; }
    }

    public class Reinforce {
        struct StateActionReward {
            public CSGameState      s;
            public CSPlayerAction   a;
            public GradData         g;
            public float            r;
        }

        public double[] PolicyControl( CSReinforce rl, int EpisodeCount, double Alpha, double Gamma, int? seed = null ) {
            List<StateActionReward> episode = new List<StateActionReward>();
            List<double> returns = new List<double>();
            double[] theta = rl.InitialTheta();

            double runningMean = 0;
            double runningVar = 1;

            for ( int episodeIndex = 0; episodeIndex < EpisodeCount; episodeIndex++ ) {
                episode.Clear();
                rl.NewGame();
                do {
                    CSGameState     s = rl.GetState();
                    (CSPlayerAction a, GradData g) = rl.Think( s, theta );
                    float           r = rl.Step( a );
                    episode.Add( new StateActionReward() { s = s, a = a, g = g, r = r } );
                } while ( rl.IsTerminal() == false );

                returns.Clear();
                double G = 0;
                for ( int t = episode.Count - 1; t >= 0; t-- ) {
                    G = episode[t].r + Gamma * G;
                    returns.Add( G );
                }
                returns.Reverse();

                // Baseline: normalise returns.
                double currentMean = returns.Average();
                double currentVar  = returns.Average( g => (g - runningMean) * (g - runningMean) );
                double std         = Math.Sqrt( Math.Max( runningVar, 1e-2 ) );

                double[] update = new double[theta.Length];
                for ( int t = 0; t < episode.Count; t++ ) {
                    double adv = ( returns[t] - runningMean ) / ( std );
                    double[] grad = rl.GradLogProb( episode[t].a, episode[t].s, episode[t].g, theta );
                    for ( int i = 0; i < theta.Length; i++ ) update[i] += adv * grad[i];
                }
                for ( int i = 0; i < theta.Length; i++ ) { 
                    theta[i] += Alpha * update[i] / episode.Count; 
                }

                const double Beta = 0.05;
                runningMean += Beta * ( currentMean - runningMean );
                runningVar += Beta * ( currentVar - runningVar );
            }

            return theta;
        }
    }
}
