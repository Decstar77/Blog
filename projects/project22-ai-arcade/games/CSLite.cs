using Raylib_cs;
using System.Numerics;

namespace arcade {
    public static class CSSettings {
        public static double GameTickRate = 60;
        public static double InvGameTickRate = 1 / GameTickRate;

        public static Color WallColor = Color.Brown;
        public static Color EmptyColor = Color.White;
        public static Color Team1Color = Color.Blue;
        public static Color Team2Color = Color.Red;

        public static int   WallSize = 50;   // Grid cell size in world units. WallSize and EmptySize must match.
        public static int   EmptySize = 50;
        public static int   PlayerSize = 20; // Diameter of the player

        public static float PlayerSpeed = 150;            // World units per second (3 cells/s)
        public static float PlayerTurnSpeed = MathF.PI * 2; // Radians per second at Rotation = 1
        public static int   PlayerHealth = 100;
        public static int   BulletDamage = 25;
        public static float GunFireCooldown = 0.20f;    // One shot every X seconds
        public static float GunVariance = 1;            // One degree variance/inaccuracy for every shot ( not in radians )
        public static float BulletSpeed = 600;          // World units per second
        public static float MaxRoundTime = 120;         // Seconds before the round is called a draw
        public static float FootstepDistance = 45;      // World units walked per footstep event
    }

    public enum CSGameResult {
        PLAYING,
        WIN,    // Team 1 won
        LOSS,   // Team 2 won
        DRAW
    }

    public struct CSPlayer {
        public int      PlayerId;
        public Vector2  Position;
        public float    Orientation;
        public int      Health;
        public float    FireCooldown;
        public int      Team;
        public float    StepDistance;

        public bool     Alive => Health > 0;
        public Vector2  Forward => new Vector2( MathF.Cos( Orientation ), MathF.Sin( Orientation ) );
    }

    public enum CSEventType {
        Shot,       // PlayerId fired
        Footstep,   // PlayerId took a step
        Hit,        // PlayerId was hit (and survived)
        Death,      // PlayerId was killed
        WallHit,    // Bullet from PlayerId hit a wall at Position
        RoundEnd,   // Result just stopped being PLAYING
    }

    // Things that happened during a Step, for audio/effects. The simulation never reads these.
    public struct CSEvent {
        public CSEventType  Type;
        public int          PlayerId;
        public Vector2      Position;
    }

    public struct CSCollider {
        public Vector2 Min;
        public Vector2 Max;
    }

    public struct CSBullet {
        public Vector2  Position;
        public float    Direction;
        public int      PlayerId; // Player that fired it
    }

    public struct CSPlayerAction {
        public int      PlayerId; // Player this action is for
        public bool     MoveLeft;
        public bool     MoveRight;
        public bool     MoveUp;
        public bool     MoveDown;
        public bool     Shoot;
        public float    Rotation; // [-1, 1]
    }

    public class CSMap {
        public static char EmptyToken  = ' ';
        public static char WallToken   = 'W';
        public static char PlayerTeam1 = '1';
        public static char PlayerTeam2 = '2';

        public char[,] MapData = new char[0,0];

        public int Rows => MapData.GetLength( 0 );
        public int Cols => MapData.GetLength( 1 );

        public void ParseMapFile( string mapFile ) {
            ParseMapLines( File.ReadAllLines( mapFile ) );
        }

        // Rows are lines of text, columns are characters. Short lines are padded with empty cells.
        public void ParseMapLines( IEnumerable<string> lines ) {
            var rows = lines.Select( l => l.TrimEnd( '\r', '\n' ) ).ToList();
            while ( rows.Count > 0 && rows[^1].Trim().Length == 0 ) {
                rows.RemoveAt( rows.Count - 1 );
            }

            int cols = rows.Count == 0 ? 0 : rows.Max( r => r.Length );
            MapData = new char[rows.Count, cols];
            for ( int r = 0; r < rows.Count; r++ ) {
                for ( int c = 0; c < cols; c++ ) {
                    char t = c < rows[r].Length ? rows[r][c] : EmptyToken;
                    bool known = t == WallToken || t == PlayerTeam1 || t == PlayerTeam2;
                    MapData[r, c] = known ? t : EmptyToken;
                }
            }
        }

        public bool IsWall( int row, int col ) {
            if ( row < 0 || col < 0 || row >= Rows || col >= Cols ) return true;
            return MapData[row, col] == WallToken;
        }
    }

    public class CSLite {
        public CSMap            MapData;
        public List<CSPlayer>   Players = new List<CSPlayer>();
        public List<CSCollider> Collider = new List<CSCollider>();
        public List<CSBullet>   Bullets = new List<CSBullet>();
        public List<CSEvent>    Events = new List<CSEvent>(); // Cleared at the start of every Step

        public CSGameResult     Result { get; private set; } = CSGameResult.PLAYING;
        public int              TickCount { get; private set; } = 0;
        public float            WorldWidth => MapData.Cols * CSSettings.WallSize;
        public float            WorldHeight => MapData.Rows * CSSettings.WallSize;

        // Camera used by the last Render call, so input code can map the mouse into world space.
        public Camera2D         Camera;

        private double          LastTick = 0;
        private readonly Random Rng;
        private readonly List<CSPlayerAction> ActionOrder = new List<CSPlayerAction>();

        public CSLite( CSMap map, int? seed = null ) {
            this.MapData = map;
            Rng = seed.HasValue ? new Random( seed.Value ) : new Random();

            float cell = CSSettings.WallSize;
            for ( int r = 0; r < map.Rows; r++ ) {
                for ( int c = 0; c < map.Cols; c++ ) {
                    char t = map.MapData[r, c];
                    Vector2 cellMin = new Vector2( c * cell, r * cell );

                    // Create players
                    if ( t == CSMap.PlayerTeam1 || t == CSMap.PlayerTeam2 ) {
                        int team = t == CSMap.PlayerTeam1 ? 0 : 1;
                        Players.Add( new CSPlayer {
                            PlayerId = Players.Count,
                            Position = cellMin + new Vector2( cell / 2 ),
                            Orientation = team == 0 ? 0 : MathF.PI,
                            Health = CSSettings.PlayerHealth,
                            FireCooldown = 0,
                            Team = team,
                        } );
                    }

                    // Create colliders
                    if ( t == CSMap.WallToken ) {
                        Collider.Add( new CSCollider { Min = cellMin, Max = cellMin + new Vector2( cell ) } );
                    }
                }
            }
        }

        // Real-time step: runs as many fixed ticks as dt covers.
        public CSGameResult Step( float dt, List<CSPlayerAction> playerActions ) {
            Events.Clear();
            LastTick += dt;
            while ( LastTick >= CSSettings.InvGameTickRate ) {
                LastTick -= CSSettings.InvGameTickRate;
                Tick( playerActions );
            }
            return Result;
        }

        // Exactly one fixed tick, for agents/training.
        public CSGameResult Step( List<CSPlayerAction> playerActions ) {
            Events.Clear();
            Tick( playerActions );
            return Result;
        }

        public void Tick( List<CSPlayerAction> playerActions ) {
            if ( Result != CSGameResult.PLAYING ) return;

            float dt = (float)CSSettings.InvGameTickRate;
            TickCount++;

            // Reload ticks for everyone, even players with no action this tick.
            for ( int i = 0; i < Players.Count; i++ ) {
                CSPlayer p = Players[i];
                p.FireCooldown = MathF.Max( 0, p.FireCooldown - dt );
                Players[i] = p;
            }

            // Apply actions in a random order each tick so no player gets a first-shot advantage.
            ActionOrder.Clear();
            ActionOrder.AddRange( playerActions );
            Rng.Shuffle( System.Runtime.InteropServices.CollectionsMarshal.AsSpan( ActionOrder ) );

            foreach ( CSPlayerAction a in ActionOrder ) {
                int i = a.PlayerId;
                if ( i < 0 || i >= Players.Count ) continue;
                CSPlayer p = Players[i];
                if ( !p.Alive ) continue;

                p.Orientation = WrapAngle( p.Orientation + Math.Clamp( a.Rotation, -1, 1 ) * CSSettings.PlayerTurnSpeed * dt );

                Vector2 move = Vector2.Zero;
                if ( a.MoveLeft )  move.X -= 1;
                if ( a.MoveRight ) move.X += 1;
                if ( a.MoveUp )    move.Y -= 1;
                if ( a.MoveDown )  move.Y += 1;
                if ( move != Vector2.Zero ) {
                    move = Vector2.Normalize( move ) * CSSettings.PlayerSpeed * dt;
                    Vector2 before = p.Position;
                    p.Position = MovePlayer( p.Position, move );
                    p.StepDistance += Vector2.Distance( before, p.Position );
                    if ( p.StepDistance >= CSSettings.FootstepDistance ) {
                        p.StepDistance -= CSSettings.FootstepDistance;
                        Emit( CSEventType.Footstep, i, p.Position );
                    }
                }

                if ( a.Shoot && p.FireCooldown <= 0 ) {
                    float spread = ( (float)Rng.NextDouble() * 2 - 1 ) * CSSettings.GunVariance * MathF.PI / 180f;
                    float dir = p.Orientation + spread;
                    Vector2 muzzle = p.Position + new Vector2( MathF.Cos( dir ), MathF.Sin( dir ) ) * ( CSSettings.PlayerSize / 2f + 2 );
                    Bullets.Add( new CSBullet { Position = muzzle, Direction = dir, PlayerId = i } );
                    p.FireCooldown = CSSettings.GunFireCooldown;
                    Emit( CSEventType.Shot, i, p.Position );
                }

                Players[i] = p;
            }

            UpdateBullets( dt );
            Result = EvaluateResult();
            if ( Result != CSGameResult.PLAYING ) {
                Emit( CSEventType.RoundEnd, -1, Vector2.Zero );
            }
        }

        private void Emit( CSEventType type, int playerId, Vector2 position ) =>
            Events.Add( new CSEvent { Type = type, PlayerId = playerId, Position = position } );

        private void UpdateBullets( float dt ) {
            float radius = CSSettings.PlayerSize / 2f;

            for ( int b = Bullets.Count - 1; b >= 0; b-- ) {
                CSBullet bullet = Bullets[b];
                Vector2 start = bullet.Position;
                Vector2 end = start + new Vector2( MathF.Cos( bullet.Direction ), MathF.Sin( bullet.Direction ) ) * CSSettings.BulletSpeed * dt;

                // Earliest wall hit along this tick's path.
                float wallT = float.MaxValue;
                foreach ( var c in Collider ) {
                    float t = SegmentBoxEntry( start, end, c );
                    if ( t >= 0 && t < wallT ) wallT = t;
                }

                // Earliest enemy hit, only counts if it happens before the wall.
                int hitPlayer = -1;
                float hitT = float.MaxValue;
                int ownerTeam = Players[bullet.PlayerId].Team;
                for ( int i = 0; i < Players.Count; i++ ) {
                    CSPlayer p = Players[i];
                    if ( !p.Alive || p.Team == ownerTeam ) continue;
                    float t = SegmentCircleEntry( start, end, p.Position, radius );
                    if ( t >= 0 && t < hitT ) { hitT = t; hitPlayer = i; }
                }

                if ( hitPlayer >= 0 && hitT <= wallT ) {
                    CSPlayer p = Players[hitPlayer];
                    p.Health = Math.Max( 0, p.Health - CSSettings.BulletDamage );
                    Players[hitPlayer] = p;
                    Emit( p.Alive ? CSEventType.Hit : CSEventType.Death, hitPlayer, p.Position );
                    Bullets.RemoveAt( b );
                    continue;
                }

                bool outOfWorld = end.X < 0 || end.Y < 0 || end.X > WorldWidth || end.Y > WorldHeight;
                if ( wallT != float.MaxValue || outOfWorld ) {
                    if ( wallT != float.MaxValue ) {
                        Emit( CSEventType.WallHit, bullet.PlayerId, Vector2.Lerp( start, end, wallT ) );
                    }
                    Bullets.RemoveAt( b );
                    continue;
                }

                bullet.Position = end;
                Bullets[b] = bullet;
            }
        }

        private CSGameResult EvaluateResult() {
            bool team1Alive = Players.Any( p => p.Alive && p.Team == 0 );
            bool team2Alive = Players.Any( p => p.Alive && p.Team == 1 );

            if ( !team1Alive && !team2Alive ) return CSGameResult.DRAW;
            if ( !team2Alive ) return CSGameResult.WIN;
            if ( !team1Alive ) return CSGameResult.LOSS;
            if ( TickCount * CSSettings.InvGameTickRate >= CSSettings.MaxRoundTime ) return CSGameResult.DRAW;
            return CSGameResult.PLAYING;
        }

        // Axis-separated movement so players slide along walls instead of sticking.
        private Vector2 MovePlayer( Vector2 pos, Vector2 delta ) {
            float half = CSSettings.PlayerSize / 2f;

            pos.X += delta.X;
            foreach ( var c in Collider ) {
                if ( !Overlaps( pos, half, c ) ) continue;
                pos.X = delta.X > 0 ? c.Min.X - half : c.Max.X + half;
            }

            pos.Y += delta.Y;
            foreach ( var c in Collider ) {
                if ( !Overlaps( pos, half, c ) ) continue;
                pos.Y = delta.Y > 0 ? c.Min.Y - half : c.Max.Y + half;
            }

            pos.X = Math.Clamp( pos.X, half, WorldWidth - half );
            pos.Y = Math.Clamp( pos.Y, half, WorldHeight - half );
            return pos;
        }

        private static bool Overlaps( Vector2 pos, float half, CSCollider c ) =>
            pos.X + half > c.Min.X && pos.X - half < c.Max.X &&
            pos.Y + half > c.Min.Y && pos.Y - half < c.Max.Y;

        // ------------------------------------------------------------------ Queries (used by bots)

        public bool HasLineOfSight( Vector2 from, Vector2 to ) {
            foreach ( var c in Collider ) {
                if ( SegmentBoxEntry( from, to, c ) >= 0 ) return false;
            }
            return true;
        }

        public (int Row, int Col) CellOf( Vector2 pos ) =>
            ( (int)( pos.Y / CSSettings.WallSize ), (int)( pos.X / CSSettings.WallSize ) );

        public Vector2 CellCenter( int row, int col ) =>
            new Vector2( ( col + 0.5f ) * CSSettings.WallSize, ( row + 0.5f ) * CSSettings.WallSize );

        public static float WrapAngle( float a ) {
            while ( a > MathF.PI )   a -= MathF.PI * 2;
            while ( a < -MathF.PI )  a += MathF.PI * 2;
            return a;
        }

        // Parametric t in [0,1] where segment a->b enters the box, or -1 if it misses.
        public static float SegmentBoxEntry( Vector2 a, Vector2 b, CSCollider box ) {
            Vector2 d = b - a;
            float tMin = 0, tMax = 1;

            for ( int axis = 0; axis < 2; axis++ ) {
                float o  = axis == 0 ? a.X : a.Y;
                float dd = axis == 0 ? d.X : d.Y;
                float lo = axis == 0 ? box.Min.X : box.Min.Y;
                float hi = axis == 0 ? box.Max.X : box.Max.Y;

                if ( MathF.Abs( dd ) < 1e-6f ) {
                    if ( o < lo || o > hi ) return -1;
                    continue;
                }
                float t1 = ( lo - o ) / dd;
                float t2 = ( hi - o ) / dd;
                if ( t1 > t2 ) ( t1, t2 ) = ( t2, t1 );
                tMin = MathF.Max( tMin, t1 );
                tMax = MathF.Min( tMax, t2 );
                if ( tMin > tMax ) return -1;
            }
            return tMin;
        }

        // Parametric t in [0,1] where segment a->b first touches the circle, or -1 if it misses.
        public static float SegmentCircleEntry( Vector2 a, Vector2 b, Vector2 center, float radius ) {
            Vector2 d = b - a;
            Vector2 f = a - center;
            float A = Vector2.Dot( d, d );
            float C = Vector2.Dot( f, f ) - radius * radius;
            if ( C <= 0 ) return 0; // Already inside
            if ( A < 1e-9f ) return -1;

            float B = 2 * Vector2.Dot( f, d );
            float disc = B * B - 4 * A * C;
            if ( disc < 0 ) return -1;

            float t = ( -B - MathF.Sqrt( disc ) ) / ( 2 * A );
            return t >= 0 && t <= 1 ? t : -1;
        }

        // ------------------------------------------------------------------ Rendering

        public void Render( float dt, List<CSPlayerAction> playerActions ) {
            int sw = Raylib.GetScreenWidth();
            int sh = Raylib.GetScreenHeight();
            float cell = CSSettings.WallSize;

            float zoom = MathF.Min( sw / WorldWidth, sh / WorldHeight ) * 0.95f;
            Camera = new Camera2D {
                Offset = new Vector2( sw / 2f, sh / 2f ),
                Target = new Vector2( WorldWidth / 2f, WorldHeight / 2f ),
                Rotation = 0,
                Zoom = zoom,
            };

            Raylib.BeginMode2D( Camera );

            Raylib.DrawRectangleV( Vector2.Zero, new Vector2( WorldWidth, WorldHeight ), CSSettings.EmptyColor );
            foreach ( var c in Collider ) {
                Raylib.DrawRectangleV( c.Min, c.Max - c.Min, CSSettings.WallColor );
            }
            // Faint grid so movement reads clearly.
            Color gridColor = Raylib.Fade( Color.LightGray, 0.5f );
            for ( int r = 0; r <= MapData.Rows; r++ ) Raylib.DrawLineV( new Vector2( 0, r * cell ), new Vector2( WorldWidth, r * cell ), gridColor );
            for ( int c = 0; c <= MapData.Cols; c++ ) Raylib.DrawLineV( new Vector2( c * cell, 0 ), new Vector2( c * cell, WorldHeight ), gridColor );

            foreach ( var b in Bullets ) {
                Vector2 dir = new Vector2( MathF.Cos( b.Direction ), MathF.Sin( b.Direction ) );
                Raylib.DrawLineEx( b.Position - dir * 8, b.Position, 3, Color.Orange );
            }

            float radius = CSSettings.PlayerSize / 2f;
            for ( int i = 0; i < Players.Count; i++ ) {
                CSPlayer p = Players[i];
                Color teamColor = p.Team == 0 ? CSSettings.Team1Color : CSSettings.Team2Color;

                if ( !p.Alive ) {
                    Raylib.DrawCircleV( p.Position, radius, Raylib.Fade( teamColor, 0.25f ) );
                    Raylib.DrawLineEx( p.Position - new Vector2( radius ), p.Position + new Vector2( radius ), 3, Color.DarkGray );
                    Raylib.DrawLineEx( p.Position + new Vector2( radius, -radius ), p.Position + new Vector2( -radius, radius ), 3, Color.DarkGray );
                    continue;
                }

                Raylib.DrawCircleV( p.Position, radius, teamColor );
                Raylib.DrawLineEx( p.Position, p.Position + p.Forward * ( radius + 10 ), 4, Color.Black );

                // Health bar
                float w = CSSettings.PlayerSize * 1.6f;
                Vector2 barPos = p.Position + new Vector2( -w / 2, -radius - 10 );
                float frac = p.Health / (float)CSSettings.PlayerHealth;
                Raylib.DrawRectangleV( barPos, new Vector2( w, 4 ), Color.DarkGray );
                Raylib.DrawRectangleV( barPos, new Vector2( w * frac, 4 ), frac > 0.5f ? Color.Green : frac > 0.25f ? Color.Yellow : Color.Red );

                // Reload indicator
                if ( p.FireCooldown > 0 ) {
                    float reload = 1 - p.FireCooldown / CSSettings.GunFireCooldown;
                    Raylib.DrawRectangleV( barPos + new Vector2( 0, 5 ), new Vector2( w * reload, 2 ), Color.Gray );
                }
            }

            Raylib.EndMode2D();

            // HUD
            for ( int i = 0; i < Players.Count; i++ ) {
                CSPlayer p = Players[i];
                Color teamColor = p.Team == 0 ? CSSettings.Team1Color : CSSettings.Team2Color;
                Raylib.DrawText( $"P{i} (team {p.Team + 1})  HP {p.Health}", 12, 12 + i * 22, 20, teamColor );
            }
            double timeLeft = Math.Max( 0, CSSettings.MaxRoundTime - TickCount * CSSettings.InvGameTickRate );
            string timer = $"{(int)timeLeft / 60}:{(int)timeLeft % 60:00}";
            Style.DrawTextCentered( timer, new Vector2( sw / 2f, 20 ), 24, Style.Text );

            if ( Result != CSGameResult.PLAYING ) {
                string msg = Result switch {
                    CSGameResult.WIN  => "TEAM 1 WINS",
                    CSGameResult.LOSS => "TEAM 2 WINS",
                    _                 => "DRAW",
                };
                Raylib.DrawRectangle( 0, sh / 2 - 50, sw, 100, Raylib.Fade( Color.Black, 0.6f ) );
                Style.DrawTextCentered( msg, new Vector2( sw / 2f, sh / 2f - 10 ), 40, Color.White );
                Style.DrawTextCentered( "Press R to restart", new Vector2( sw / 2f, sh / 2f + 25 ), 20, Color.LightGray );
            }
        }
    }
}
