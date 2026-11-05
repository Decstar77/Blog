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

        public static int WallSize = 50;
        public static int EmptySize = 50;
        public static int PlayerSize = 20;

        public static float PlayerSpeed = 0; // TODO: Pick sensible values
        public static int   PlayerHealth = 100;
        public static float GunFireCooldown = 1.0f; // One shot every X seconds
        public static float GunVariance = 1;        // One degree variance/inaccuracy for every shot ( not in radians )
        public static float BulletSpeed = 0; // TODO: Pick sensible values
    }

    public enum CSGameResult { 
        WIN,
        LOSS,
        DRAW
    }

    public struct CSPlayer {
        public Vector2  Position;
        public float    Orientation;
        public int      Health;
        public float    FireCooldown;
    }

    public struct CSCollider {
        public Vector2 Min;
        public Vector2 Max;
    }

    public struct CSBullet {
        public Vector2  Position;
        public float    Direction;
    }

    public struct CSPlayerAction {
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

        public void ParseMapFile( string mapFile ) {
            // TODO
        }
    }

    public class CSLite {
        public CSMap            MapData;
        public List<CSPlayer>   Players = new List<CSPlayer>();
        public List<CSCollider> Collider = new List<CSCollider>();
        public List<CSBullet>   Bullets = new List<CSBullet>();

        private double          LastTick = 0;

        public CSLite( CSMap map ) {
            this.MapData = map;
            // Create players
            // Create colliders
        }

        public CSGameResult Step( float dt, List<CSPlayerAction> playerActions ) {
            LastTick += dt;
            if ( LastTick >= CSSettings.GameTickRate ) {
                LastTick -= CSSettings.GameTickRate;
                Tick( playerActions );
            }
            return CSGameResult.WIN;
        }

        public CSGameResult Step( List<CSPlayerAction> playerActions ) {
            Tick( playerActions );
            return CSGameResult.WIN;
        }

        public void Tick( List<CSPlayerAction> playerActions ) {

        }

        public void Render( float dt, List<CSPlayerAction> playerActions ) {

        }
    }
}
