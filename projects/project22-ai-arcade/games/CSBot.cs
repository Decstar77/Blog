using System.Numerics;

namespace arcade {
    // Simple scripted opponent:
    //  - Can see an enemy: turn toward them, shoot once lined up, strafe and hold a comfortable range.
    //  - Can't see one: BFS over the map grid toward the nearest enemy and walk the path.
    public class CSBot {
        public float AimTolerance = 4f * MathF.PI / 180f; // Fire when within this many radians of the target
        public float PreferredRange = 220;                // Close in when farther than this
        public float StrafeSwitchTime = 1.2f;             // Seconds between strafe direction flips

        public readonly int     PlayerId;
        private readonly Random Rng;
        private int             StrafeSign = 1;
        private float           StrafeTimer = 0;

        public CSBot( int playerIndex, int? seed = null ) {
            PlayerId = playerIndex;
            Rng = seed.HasValue ? new Random( seed.Value ) : new Random();
        }

        public CSPlayerAction Think( CSLite game, float dt ) {
            var action = new CSPlayerAction { PlayerId = PlayerId };
            CSPlayer me = game.Players[PlayerId];
            if ( !me.Alive || game.Result != CSGameResult.PLAYING ) return action;

            int target = NearestEnemy( game, me );
            if ( target < 0 ) return action;
            CSPlayer enemy = game.Players[target];

            Vector2 toEnemy = enemy.Position - me.Position;
            float dist = toEnemy.Length();

            if ( game.HasLineOfSight( me.Position, enemy.Position ) ) {
                float aimError = AimAt( ref action, me, MathF.Atan2( toEnemy.Y, toEnemy.X ) );
                action.Shoot = MathF.Abs( aimError ) < AimTolerance;

                StrafeTimer -= dt;
                if ( StrafeTimer <= 0 ) {
                    StrafeSign = Rng.Next( 2 ) == 0 ? -1 : 1;
                    StrafeTimer = StrafeSwitchTime * ( 0.5f + (float)Rng.NextDouble() );
                }

                Vector2 dir = toEnemy / MathF.Max( dist, 1e-3f );
                Vector2 strafe = new Vector2( -dir.Y, dir.X ) * StrafeSign;
                Vector2 approach = dist > PreferredRange ? dir : dist < PreferredRange * 0.5f ? -dir : Vector2.Zero;
                SetMove( ref action, strafe + approach );
            }
            else {
                Vector2? waypoint = NextWaypoint( game, me.Position, enemy.Position );
                Vector2 goal = waypoint ?? enemy.Position;
                Vector2 toGoal = goal - me.Position;
                SetMove( ref action, toGoal );

                // Look where we're going so we're ready when the enemy shows up.
                if ( toGoal.LengthSquared() > 1 ) {
                    AimAt( ref action, me, MathF.Atan2( toGoal.Y, toGoal.X ) );
                }
            }

            return action;
        }

        private int NearestEnemy( CSLite game, CSPlayer me ) {
            int best = -1;
            float bestDist = float.MaxValue;
            for ( int i = 0; i < game.Players.Count; i++ ) {
                CSPlayer p = game.Players[i];
                if ( i == PlayerId || !p.Alive || p.Team == me.Team ) continue;
                float d = Vector2.DistanceSquared( p.Position, me.Position );
                if ( d < bestDist ) { bestDist = d; best = i; }
            }
            return best;
        }

        // Sets Rotation to turn toward the desired angle, returns the remaining angle error.
        private static float AimAt( ref CSPlayerAction action, CSPlayer me, float desired ) {
            float error = CSLite.WrapAngle( desired - me.Orientation );
            float maxTurnPerTick = CSSettings.PlayerTurnSpeed * (float)CSSettings.InvGameTickRate;
            action.Rotation = Math.Clamp( error / maxTurnPerTick, -1, 1 );
            return error;
        }

        private static void SetMove( ref CSPlayerAction action, Vector2 dir ) {
            const float deadZone = 2f;
            if ( dir.LengthSquared() < 1e-6f ) return;
            Vector2 n = Vector2.Normalize( dir );
            // Only press an axis if it contributes meaningfully, or if we're far off on it.
            action.MoveLeft  = n.X < -0.3f || dir.X < -deadZone && MathF.Abs( n.X ) > 0.1f;
            action.MoveRight = n.X >  0.3f || dir.X >  deadZone && MathF.Abs( n.X ) > 0.1f;
            action.MoveUp    = n.Y < -0.3f || dir.Y < -deadZone && MathF.Abs( n.Y ) > 0.1f;
            action.MoveDown  = n.Y >  0.3f || dir.Y >  deadZone && MathF.Abs( n.Y ) > 0.1f;
        }

        private static readonly int[] DR = { -1, 1, 0, 0 };
        private static readonly int[] DC = { 0, 0, -1, 1 };

        // Centre of the next cell on a shortest 4-connected grid path, or null if no path exists.
        public static Vector2? NextWaypoint( CSLite game, Vector2 from, Vector2 to ) {
            CSMap map = game.MapData;
            var start = game.CellOf( from );
            var goal = game.CellOf( to );
            if ( start == goal ) return to;

            var parent = new (int Row, int Col)[map.Rows, map.Cols];
            var seen = new bool[map.Rows, map.Cols];
            var queue = new Queue<(int Row, int Col)>();
            queue.Enqueue( start );
            seen[start.Row, start.Col] = true;

            bool found = false;
            while ( queue.Count > 0 ) {
                var cur = queue.Dequeue();
                if ( cur == goal ) { found = true; break; }
                for ( int k = 0; k < 4; k++ ) {
                    int r = cur.Row + DR[k], c = cur.Col + DC[k];
                    if ( map.IsWall( r, c ) || seen[r, c] ) continue;
                    seen[r, c] = true;
                    parent[r, c] = cur;
                    queue.Enqueue( ( r, c ) );
                }
            }
            if ( !found ) return null;

            // Walk back from the goal to the cell right after the start.
            var step = goal;
            while ( parent[step.Row, step.Col] != start ) {
                step = parent[step.Row, step.Col];
            }
            return game.CellCenter( step.Row, step.Col );
        }
    }
}
