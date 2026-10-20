using System.Numerics;
using Raylib_cs;

namespace arcade {
    // Shared palette and layout helpers so every game draws in the same visual language.
    public static class Style
    {
        public static readonly Color Background = new Color(24, 26, 33, 255);
        public static readonly Color Panel      = new Color(36, 39, 49, 255);
        public static readonly Color PanelEdge  = new Color(58, 62, 76, 255);
        public static readonly Color Text       = new Color(226, 228, 235, 255);
        public static readonly Color TextDim    = new Color(140, 145, 160, 255);

        // Player 0 and player 1.
        public static readonly Color P0         = new Color(236, 94, 94, 255);
        public static readonly Color P1         = new Color(250, 200, 70, 255);
        public static readonly Color Highlight  = new Color(120, 220, 140, 255);

        public static Color PlayerColor(int p) => p == 0 ? P0 : P1;

        // Largest rect with the given cols:rows aspect that fits in area (minus padding), centred.
        public static Rectangle FitGrid(Rectangle area, int cols, int rows, float padding = 16)
        {
            float w = area.Width - 2 * padding;
            float h = area.Height - 2 * padding;
            float cell = MathF.Min(w / cols, h / rows);
            float gw = cell * cols, gh = cell * rows;
            return new Rectangle(area.X + (area.Width - gw) / 2, area.Y + (area.Height - gh) / 2, gw, gh);
        }

        public static Rectangle Inflate(Rectangle r, float by) =>
            new Rectangle(r.X - by, r.Y - by, r.Width + 2 * by, r.Height + 2 * by);

        public static Vector2 Center(Rectangle r) => new Vector2(r.X + r.Width / 2, r.Y + r.Height / 2);

        public static void DrawTextCentered(string text, Vector2 center, int size, Color color)
        {
            int w = Raylib.MeasureText(text, size);
            Raylib.DrawText(text, (int)(center.X - w / 2f), (int)(center.Y - size / 2f), size, color);
        }
    }
}
