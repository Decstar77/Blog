// Classic 3x3 noughts and crosses. Two players alternate placing marks; first to get three in a
// row wins. Small enough to solve exactly with minimax, which makes it a good sanity check for agents.
// Actions are cell indices 0..8, row-major. Player 0 is X and moves first.

using System.Numerics;
using Raylib_cs;

namespace arcade {
    public class TicTacToe : ArcadeGame
    {
        private static readonly int[][] Lines =
        {
            new[] { 0, 1, 2 }, new[] { 3, 4, 5 }, new[] { 6, 7, 8 },
            new[] { 0, 3, 6 }, new[] { 1, 4, 7 }, new[] { 2, 5, 8 },
            new[] { 0, 4, 8 }, new[] { 2, 4, 6 },
        };

        // -1 = empty, otherwise the player index that owns the cell.
        private int[]   board = Enumerable.Repeat(-1, 9).ToArray();
        private int     player = 0;
        private int     winner = -1;
        private int     moves = 0;

        public int NumPlayers => 2;
        public int NumActions => 9;

        public int CurrentPlayer() => player;

        public int[] LegalActions()
        {
            if (IsTerminal()) return Array.Empty<int>();
            return Enumerable.Range(0, 9).Where(i => board[i] == -1).ToArray();
        }

        public void Apply(int action)
        {
            if (IsTerminal() || action < 0 || action >= 9 || board[action] != -1)
                throw new ArgumentException($"Illegal move {action}");

            board[action] = player;
            moves++;
            foreach (var line in Lines)
            {
                if (line.All(i => board[i] == player))
                {
                    winner = player;
                    break;
                }
            }
            player = 1 - player;
        }

        public bool IsTerminal() => winner != -1 || moves == 9;

        public float[] Returns()
        {
            if (winner == -1) return new float[] { 0, 0 };
            return winner == 0 ? new float[] { 1, -1 } : new float[] { -1, 1 };
        }

        public ArcadeGame Clone()
        {
            var copy = (TicTacToe)MemberwiseClone();
            copy.board = (int[])board.Clone();
            return copy;
        }

        public void Render(Rectangle area)
        {
            Rectangle grid = Style.FitGrid(area, 3, 3, 32);
            float cell = grid.Width / 3;
            float thick = MathF.Max(3, cell * 0.05f);

            Raylib.DrawRectangleRounded(Style.Inflate(grid, 16), 0.06f, 8, Style.Panel);

            // Inner grid lines only, like a hand-drawn board.
            for (int i = 1; i < 3; i++)
            {
                float x = grid.X + i * cell, y = grid.Y + i * cell;
                Raylib.DrawLineEx(new Vector2(x, grid.Y), new Vector2(x, grid.Y + grid.Height), thick, Style.PanelEdge);
                Raylib.DrawLineEx(new Vector2(grid.X, y), new Vector2(grid.X + grid.Width, y), thick, Style.PanelEdge);
            }

            for (int i = 0; i < 9; i++)
            {
                if (board[i] == -1) continue;
                Vector2 c = CellCenter(grid, i);
                float r = cell * 0.3f;
                if (board[i] == 0)
                {
                    Raylib.DrawLineEx(c + new Vector2(-r, -r), c + new Vector2(r, r), thick * 1.6f, Style.P0);
                    Raylib.DrawLineEx(c + new Vector2(-r, r), c + new Vector2(r, -r), thick * 1.6f, Style.P0);
                }
                else
                {
                    Raylib.DrawRing(c, r - thick * 0.8f, r + thick * 0.8f, 0, 360, 48, Style.P1);
                }
            }

            if (winner != -1)
            {
                var line = Lines.First(l => l.All(i => board[i] == winner));
                Vector2 a = CellCenter(grid, line[0]), b = CellCenter(grid, line[2]);
                Vector2 ext = Vector2.Normalize(b - a) * cell * 0.4f;
                Raylib.DrawLineEx(a - ext, b + ext, thick * 1.4f, Style.Highlight);
            }
        }

        private static Vector2 CellCenter(Rectangle grid, int i)
        {
            float cell = grid.Width / 3;
            return new Vector2(grid.X + (i % 3 + 0.5f) * cell, grid.Y + (i / 3 + 0.5f) * cell);
        }

        public override string ToString()
        {
            var sb = new System.Text.StringBuilder();
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int v = board[r * 3 + c];
                    sb.Append(v == -1 ? '.' : v == 0 ? 'X' : 'O');
                }
                sb.AppendLine();
            }
            return sb.ToString();
        }
    }
}
