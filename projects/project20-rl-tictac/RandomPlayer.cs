
namespace rl
{
    public class RandomPlayer
    {
        private Random random = new Random();
        private int piece = 0;
        
        public void NewGame(int piece)
        {
            this.piece = piece;
        }

        public GameMove GetNextMove(TicTacToe game)
        {
            int x;
            int y;
            do
            {
                x = random.Next(3);
                y = random.Next(3);
            } while (game.State[x, y] != TicTacToe.Empty);

            GameMove move = new GameMove();
            move.x = x;
            move.y = y;
            move.p = piece;
            return move;
        }
    }
}
