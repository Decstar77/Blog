using System.Numerics;
using Raylib_cs;

namespace arcade {
    public class FrozenLake 
    {
        public static readonly string[] Map4x4 =
        {
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG",
        };

        public static readonly string[] Map8x8 =
        {
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        };

        protected static readonly int[] DRow = { 0, 1, 0, -1 };
        protected static readonly int[] DCol = { -1, 0, 1, 0 };

        protected readonly string[]   map;
        protected readonly bool       slippery;
        protected readonly int        maxSteps;
        protected Random              rng;

        protected int     row = 0;
        protected int     col = 0;
        protected int     steps = 0;

        // Last intended action and the direction actually taken, so Render can show slips.
        protected int     lastAction = -1;
        protected int     lastDir = -1;

        public FrozenLake(string[]? map = null, bool slippery = true, int maxSteps = 100, int? seed = null)
        {
            this.map = map ?? Map4x4;
            this.slippery = slippery;
            this.maxSteps = maxSteps;
            rng = seed.HasValue ? new Random(seed.Value) : new Random();

            for (int r = 0; r < this.map.Length; r++)
            {
                int c = this.map[r].IndexOf('S');
                if (c >= 0) { row = r; col = c; }
            }
        }

        public int NumPlayers => 1;
        public int NumActions => 4;

        public int Rows => map.Length;
        public int Cols => map[0].Length;

        // Flat cell index, handy as a tabular-RL state id.
        public int State => row * Cols + col;

        private char Tile => map[row][col];

        public int CurrentPlayer() => 0;

        public int[] LegalActions() => IsTerminal() ? Array.Empty<int>() : new[] { 0, 1, 2, 3 };

        public void Apply(int action)
        {
            if (IsTerminal() || action < 0 || action >= 4)
                throw new ArgumentException($"Illegal move {action}");

            int dir = action;
            if (slippery)
            {
                // 0: as intended, 1: rotate one way, 2: rotate the other.
                int roll = rng.Next(3);
                if (roll == 1) dir = (action + 1) % 4;
                else if (roll == 2) dir = (action + 3) % 4;
            }

            (row, col) = Move(row, col, dir);
            steps++;
            lastAction = action;
            lastDir = dir;
        }

        public bool IsTerminal() => Tile == 'H' || Tile == 'G' || steps >= maxSteps;

        public float[] Returns() => new float[] { Tile == 'G' ? 1 : 0 };

        protected (int Row, int Col) Move(int r, int c, int dir) => (Math.Clamp(r + DRow[dir], 0, Rows - 1), Math.Clamp(c + DCol[dir], 0, Cols - 1));

        private static readonly Color Ice       = new Color(178, 216, 238, 255);
        private static readonly Color IceShine  = new Color(214, 236, 249, 255);
        private static readonly Color Water     = new Color(18, 40, 72, 255);
        private static readonly Color WaterEdge = new Color(70, 120, 170, 255);
        private static readonly Color Slip      = new Color(255, 140, 60, 255);

        public void Render(Rectangle area)
        {
            Rectangle grid = Style.FitGrid(area, Cols, Rows, 24);
            float cell = grid.Width / Cols;
            float gap = MathF.Max(2, cell * 0.04f);

            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Cols; c++)
                {
                    var tile = new Rectangle(grid.X + c * cell + gap, grid.Y + r * cell + gap, cell - 2 * gap, cell - 2 * gap);
                    Vector2 mid = Style.Center(tile);
                    char t = map[r][c];

                    Raylib.DrawRectangleRounded(tile, 0.15f, 6, Ice);
                    // A diagonal glint so the ice doesn't look flat.
                    if (t != 'G')
                        Raylib.DrawLineEx(new Vector2(tile.X + tile.Width * 0.2f, tile.Y + tile.Height * 0.45f),
                                          new Vector2(tile.X + tile.Width * 0.45f, tile.Y + tile.Height * 0.2f),
                                          MathF.Max(2, cell * 0.04f), IceShine);

                    if (t == 'H')
                    {
                        Raylib.DrawEllipseV(mid, tile.Width * 0.4f, tile.Height * 0.34f, WaterEdge);
                        Raylib.DrawEllipseV(mid, tile.Width * 0.34f, tile.Height * 0.28f, Water);
                    }
                    else if (t == 'G')
                    {
                        DrawFlag(tile);
                    }
                    else if (t == 'S')
                    {
                        Raylib.DrawRectangleRoundedLinesEx(Style.Inflate(tile, -cell * 0.08f), 0.15f, 6, MathF.Max(2, cell * 0.03f), WaterEdge);
                    }
                }
            }

            DrawAgent(grid, cell);
        }

        private static void DrawFlag(Rectangle tile)
        {
            float w = tile.Width, h = tile.Height;
            var baseP = new Vector2(tile.X + w * 0.35f, tile.Y + h * 0.82f);
            var top = new Vector2(tile.X + w * 0.35f, tile.Y + h * 0.18f);
            Raylib.DrawEllipseV(baseP, w * 0.16f, h * 0.05f, Raylib.ColorBrightness(Ice, -0.2f));
            Raylib.DrawLineEx(baseP, top, MathF.Max(2, w * 0.04f), Style.PanelEdge);
            // Counter-clockwise winding, as raylib requires.
            Raylib.DrawTriangle(top, new Vector2(top.X, top.Y + h * 0.28f), new Vector2(top.X + w * 0.38f, top.Y + h * 0.14f), Style.P1);
        }

        private void DrawAgent(Rectangle grid, float cell)
        {
            var p = new Vector2(grid.X + (col + 0.5f) * cell, grid.Y + (row + 0.5f) * cell);
            float r = cell * 0.24f;
            char t = Tile;

            if (t == 'H')
            {
                // Fallen in: just the top of the head poking out of the water.
                Raylib.DrawCircleSector(p, r, 180, 360, 24, Style.Text);
                return;
            }

            Raylib.DrawCircleV(p + new Vector2(0, r * 0.15f), r, Raylib.Fade(Style.Background, 0.25f));
            Raylib.DrawCircleV(p, r, t == 'G' ? Style.Highlight : Style.P0);
            Raylib.DrawCircleV(p, r * 0.75f, Raylib.ColorBrightness(t == 'G' ? Style.Highlight : Style.P0, 0.25f));

            if (lastDir == -1) return;

            // Arrow for the direction actually moved; orange when the ice sent us somewhere unintended.
            var d = new Vector2(DCol[lastDir], DRow[lastDir]);
            var side = new Vector2(-d.Y, d.X);
            Vector2 tip = p + d * r * 0.6f;
            Vector2 back = p - d * r * 0.25f;
            Color c = lastDir == lastAction ? Style.Background : Slip;
            Vector2 a = back + side * r * 0.4f, b = back - side * r * 0.4f;
            // Pick the winding that's counter-clockwise on screen (y points down).
            if (Cross(a - tip, b - tip) > 0) (a, b) = (b, a);
            Raylib.DrawTriangle(tip, a, b, c);
        }

        private static float Cross(Vector2 u, Vector2 v) => u.X * v.Y - u.Y * v.X;

        public override string ToString()
        {
            var sb = new System.Text.StringBuilder();
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Cols; c++)
                    sb.Append(r == row && c == col ? '@' : map[r][c]);
                sb.AppendLine();
            }
            return sb.ToString();
        }
    }
}
