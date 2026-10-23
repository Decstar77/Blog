// The natural step up from TicTacToe. Its state space is much larger (~4.5 trillion positions), so
// exhaustive minimax is out and agents need depth limits, heuristics, or MCTS.
// 7 columns x 6 rows. Actions are column indices 0..6; the piece drops to the lowest free row.
// Player 0 moves first. First to connect four horizontally, vertically or diagonally wins.

using System.Numerics;
using Raylib_cs;

namespace arcade {
    public class ConnectFour : ArcadeGame
    {
        public const int Cols = 7;
        public const int Rows = 6;

        // board[row, col], row 0 is the bottom. -1 = empty, otherwise the owning player.
        private int[,]  board = new int[Rows, Cols];
        private int[]   heights = new int[Cols];
        private int     player = 0;
        private int     winner = -1;
        private int     moves = 0;
        private int     lastRow = -1;
        private int     lastCol = -1;

        public ConnectFour()
        {
            for (int r = 0; r < Rows; r++)
                for (int c = 0; c < Cols; c++)
                    board[r, c] = -1;
        }

        public int NumPlayers => 2;
        public int NumActions => Cols;

        public int CurrentPlayer() => player;

        public int[] LegalActions()
        {
            if (IsTerminal()) return Array.Empty<int>();
            return Enumerable.Range(0, Cols).Where(c => heights[c] < Rows).ToArray();
        }

        public void Apply(int action)
        {
            if (IsTerminal() || action < 0 || action >= Cols || heights[action] >= Rows)
                throw new ArgumentException($"Illegal move {action}");

            int row = heights[action]++;
            board[row, action] = player;
            moves++;
            lastRow = row;
            lastCol = action;
            if (MakesFour(row, action)) winner = player;
            player = 1 - player;
        }

        // Only the newly placed piece can complete a line, so check the four directions through it.
        private bool MakesFour(int row, int col)
        {
            int p = board[row, col];
            int[,] dirs = { { 0, 1 }, { 1, 0 }, { 1, 1 }, { 1, -1 } };
            for (int d = 0; d < 4; d++)
            {
                int count = 1 + Count(row, col, dirs[d, 0], dirs[d, 1], p)
                              + Count(row, col, -dirs[d, 0], -dirs[d, 1], p);
                if (count >= 4) return true;
            }
            return false;
        }

        private int Count(int row, int col, int dr, int dc, int p)
        {
            int n = 0;
            for (int r = row + dr, c = col + dc;
                 r >= 0 && r < Rows && c >= 0 && c < Cols && board[r, c] == p;
                 r += dr, c += dc)
                n++;
            return n;
        }

        // Heuristic score from player 0's perspective, for depth-limited search. Every 4-cell window
        // that only one player occupies counts toward that player, more for 3 than for 2, and pieces
        // in the center column get a small bonus since they take part in the most lines.
        public int Evaluate()
        {
            int score = 0;
            int[,] dirs = { { 0, 1 }, { 1, 0 }, { 1, 1 }, { 1, -1 } };
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Cols; c++)
                {
                    if (c == Cols / 2 && board[r, c] != -1) score += board[r, c] == 0 ? 3 : -3;
                    for (int d = 0; d < 4; d++)
                    {
                        int er = r + 3 * dirs[d, 0], ec = c + 3 * dirs[d, 1];
                        if (er < 0 || er >= Rows || ec < 0 || ec >= Cols) continue;
                        int mine = 0, theirs = 0;
                        for (int k = 0; k < 4; k++)
                        {
                            int v = board[r + k * dirs[d, 0], c + k * dirs[d, 1]];
                            if (v == 0) mine++;
                            else if (v == 1) theirs++;
                        }
                        if (theirs == 0) score += WindowScore(mine);
                        else if (mine == 0) score -= WindowScore(theirs);
                    }
                }
            }
            return score;
        }

        private static int WindowScore(int count) => count switch { 3 => 50, 2 => 5, _ => 0 };

        public bool IsTerminal() => winner != -1 || moves == Rows * Cols;

        public float[] Returns()
        {
            if (winner == -1) return new float[] { 0, 0 };
            return winner == 0 ? new float[] { 1, -1 } : new float[] { -1, 1 };
        }

        public ArcadeGame Clone()
        {
            var copy = (ConnectFour)MemberwiseClone();
            copy.board = (int[,])board.Clone();
            copy.heights = (int[])heights.Clone();
            return copy;
        }

        private static readonly Color BoardColor = new Color(48, 86, 170, 255);
        private static readonly Color BoardEdge  = new Color(34, 64, 132, 255);

        public void Render(Rectangle area)
        {
            Rectangle grid = Style.FitGrid(area, Cols, Rows, 32);
            float cell = grid.Width / Cols;
            float radius = cell * 0.4f;

            Rectangle frame = Style.Inflate(grid, cell * 0.15f);
            Raylib.DrawRectangleRounded(frame, 0.05f, 8, BoardColor);
            Raylib.DrawRectangleRoundedLinesEx(frame, 0.05f, 8, 3, BoardEdge);

            var winning = WinningCells();
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Cols; c++)
                {
                    Vector2 p = CellCenter(grid, r, c);
                    int v = board[r, c];
                    if (v == -1)
                    {
                        Raylib.DrawCircleV(p, radius, Style.Background);
                        continue;
                    }

                    // Darker rim plus a lighter inner disc gives the counters a bit of depth.
                    Color col = Style.PlayerColor(v);
                    Raylib.DrawCircleV(p, radius, Raylib.ColorBrightness(col, -0.3f));
                    Raylib.DrawCircleV(p, radius * 0.86f, col);
                    Raylib.DrawRing(p, radius * 0.55f, radius * 0.62f, 0, 360, 32, Raylib.ColorBrightness(col, -0.15f));

                    if (winning.Contains((r, c)))
                        Raylib.DrawRing(p, radius * 0.92f, radius * 1.08f, 0, 360, 48, Style.Highlight);
                    else if (r == lastRow && c == lastCol)
                        Raylib.DrawCircleV(p, radius * 0.15f, Raylib.ColorBrightness(col, 0.5f));
                }
            }
        }

        private static Vector2 CellCenter(Rectangle grid, int row, int col)
        {
            float cell = grid.Width / Cols;
            return new Vector2(grid.X + (col + 0.5f) * cell, grid.Y + (Rows - 1 - row + 0.5f) * cell);
        }

        // Every cell that's part of a line of four or more, for highlighting at game end.
        private HashSet<(int, int)> WinningCells()
        {
            var cells = new HashSet<(int, int)>();
            if (winner == -1) return cells;
            int[,] dirs = { { 0, 1 }, { 1, 0 }, { 1, 1 }, { 1, -1 } };
            for (int r = 0; r < Rows; r++)
                for (int c = 0; c < Cols; c++)
                    for (int d = 0; d < 4; d++)
                    {
                        int dr = dirs[d, 0], dc = dirs[d, 1];
                        int er = r + 3 * dr, ec = c + 3 * dc;
                        if (er < 0 || er >= Rows || ec < 0 || ec >= Cols) continue;
                        bool four = true;
                        for (int k = 0; k < 4 && four; k++)
                            four = board[r + k * dr, c + k * dc] == winner;
                        if (four)
                            for (int k = 0; k < 4; k++)
                                cells.Add((r + k * dr, c + k * dc));
                    }
            return cells;
        }

        public override string ToString()
        {
            var sb = new System.Text.StringBuilder();
            for (int r = Rows - 1; r >= 0; r--)
            {
                for (int c = 0; c < Cols; c++)
                {
                    int v = board[r, c];
                    sb.Append(v == -1 ? '.' : v == 0 ? 'X' : 'O');
                }
                sb.AppendLine();
            }
            sb.AppendLine("0123456");
            return sb.ToString();
        }
    }
}
