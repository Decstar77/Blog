
using ShellProgressBar;

namespace rl {
    public class Program {
        public static Random random = new Random();
        public static ProgressBarOptions options = new ProgressBarOptions {
            ProgressCharacter = '=',
            ProgressBarOnBottom = true
        };

        public static GameMove MaxAction( double[,,] Q, TicTacToe game, int state, int p ) {
            GameMove action = new GameMove();
            double maxValue = double.MinValue;
            for ( int x = 0; x < 3; x++ ) {
                for ( int y = 0; y < 3; y++ ) {
                    if ( game.State[x, y] == TicTacToe.Empty ) {
                        double value = Q[state, x, y];
                        if ( value > maxValue ) {
                            maxValue = value;
                            action.x = x;
                            action.y = y;
                            action.p = p;
                        }
                    }
                }
            }
            return action;
        }

        public static GameMove ChooseAction( double[,,] Q, TicTacToe game, int state, int p ) {
            double epsilon = 0.1;
            if ( random.NextDouble() > epsilon ) {
                return MaxAction( Q, game, state, p );
            }
            else {
                int x;
                int y;
                do {
                    x = random.Next( 3 );
                    y = random.Next( 3 );
                } while ( game.State[x, y] != TicTacToe.Empty );

                GameMove action = new GameMove();
                action.x = x;
                action.y = y;
                action.p = p;
                return action;
            }
        }

        public static void QLearning( int episodeCount ) {
            double alpha = 0.1;
            double[,,] Q = new double[20_000, 3, 3];

            TicTacToe game = new TicTacToe();
            RandomPlayer player = new RandomPlayer();
            int totalWins = 0;
            int totalLoss = 0;
            int totalDraws= 0;
            using ( var pbar = new ProgressBar( episodeCount, "Training Episodes", options ) ) {
                for ( int episodeIndex = 0; episodeIndex < episodeCount; episodeIndex++ ) {
                    game.NewGame();
                    player.NewGame( TicTacToe.O );

                    GameState status = GameState.Playing;
                    while ( status == GameState.Playing ) {
                        int currentState = game.StateHash();
                        GameMove move = ChooseAction(Q, game, currentState, TicTacToe.X);
                        status = game.PlayMove( move );
                        double reward = status == GameState.Finished ? 1 : 0;
                        if ( status == GameState.Finished ) {
                            Q[currentState, move.x, move.y] = reward;
                            totalWins++;
                            break;
                        }
                        else if ( status == GameState.Draw ) {
                            Q[currentState, move.x, move.y] = 0;
                            totalDraws++;
                            break;
                        }

                        status = game.PlayMove( player.GetNextMove( game ) );
                        if ( status == GameState.Finished ) {
                            totalLoss++;
                            reward = -1;
                        } else if ( status == GameState.Draw ) {
                            totalDraws++;
                        }

                        int nextState = game.StateHash();
                        GameMove maxAction = MaxAction(Q, game, nextState, TicTacToe.X);
                        Q[currentState, move.x, move.y] += alpha * ( reward + Q[nextState, maxAction.x, maxAction.y] - Q[currentState, move.x, move.y] );
                    }

                    pbar.Tick( $"Wins/Draw/Loss - {totalWins}/{totalDraws}/{totalLoss}" );
                }
            }

            Console.WriteLine();
        }

        public static void Main() {
            QLearning( 1_000_000 );
            Console.ReadLine();
        }
    }
}

