namespace rl
{
    public record GameMove
    {
        public int x;
        public int y;
        public int p;
    }

    public enum GameState
    {
        Playing,
        Finished,
        Draw
    }

    public class TicTacToe
    {
        public const int Empty = 0;
        public const int X = 1;
        public const int O = 2;
        public const int W = 3;
        public const int H = 3;
        public int[,] State = new int[W, H];
        private static readonly int[] PowersOf3 = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561 };

        public void NewGame()
        {
            for (int x = 0; x < W; x++)
            {
                for (int y = 0; y < H; y++)
                {
                    State[x, y] = Empty;
                }
            }
        }

        public int StateHash()
        {
            int hash = 0;
            for (int x = 0; x < W; x++)
            {
                for (int y = 0; y < H; y++)
                {
                    hash += State[x, y] * PowersOf3[ x + y * 3 ];
                }
            }
            return hash;
        }

        public GameState PlayMove(GameMove move)
        {
            State[move.x, move.y] = move.p;
            if (CheckForPieceWin(move.p))
            {
                return GameState.Finished;
            }
            else if (CheckForDraw())
            {
                return GameState.Draw;
            }
            else
            {
                return GameState.Playing;
            }
        }

        public bool CheckForDraw()
        {
            for (int x = 0; x < W; x++)
            {
                for (int y = 0; y < H; y++)
                {
                    if (State[x, y] == Empty) return false;
                }
            }

            return !CheckForPieceWin(X) && !CheckForPieceWin(O);
        }

        public bool CheckForPieceWin(int p)
        {
            for (int i = 0; i < 3; i++)
            {
                if (State[i, 0] == p && State[i, 1] == p && State[i, 2] == p) return true;
                if (State[0, i] == p && State[1, i] == p && State[2, i] == p) return true;
            }

            if (State[0, 0] == p && State[1, 1] == p && State[2, 2] == p) return true;
            if (State[2, 0] == p && State[1, 1] == p && State[0, 2] == p) return true;

            return false;
        }

        public void PrintBoard()
        {
            for (int y = 0; y < H; y++)
            {
                for (int x = 0; x < W; x++)
                {
                    char c = State[x, y] switch
                    {
                        X => 'X',
                        O => 'O',
                        _ => '.',
                    };
                    Console.Write(c);
                    if (x < W - 1) Console.Write(" | ");
                }
                Console.WriteLine();
                if (y < H - 1) Console.WriteLine("--+---+--");
            }
        }
    }
}