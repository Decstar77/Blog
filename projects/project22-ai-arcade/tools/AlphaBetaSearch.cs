using System;

namespace arcade {

    public static class AlphaBeta {
        public const int WinScore = 1_000_000;

        // Returns the best action for the current player, or -1 if there are no legal actions.
        //public static int Search( ArcadeGame game, int depth, Func<ArcadeGame, int> eval ) {
        //    if ( game.NumPlayers != 2 ) {
        //        throw new ArgumentException( "AlphaBeta only supports two-player games" );
        //    }

        //    bool isMax = game.CurrentPlayer() == 0;
        //    int alpha = int.MinValue;
        //    int beta = int.MaxValue;
        //    int bestAction = -1;
        //    int bestValue = 0;
        //    foreach ( int action in game.LegalActions() ) {
        //        int value = DoSearch( Child( game, action ), depth - 1, alpha, beta, eval );
        //        if ( bestAction == -1 || ( isMax ? value > bestValue : value < bestValue ) ) {
        //            bestValue = value;
        //            bestAction = action;
        //        }
        //        if ( isMax ) {
        //            alpha = Math.Max( alpha, bestValue );
        //        } else {
        //            beta = Math.Min( beta, bestValue );
        //        }
        //    }
        //    return bestAction;
        //}

        //private static int DoSearch( ArcadeGame game, int depth, int alpha, int beta, Func<ArcadeGame, int> eval ) {
        //    if ( game.IsTerminal() ) {
        //        // Adding the remaining depth makes the search prefer quicker wins and slower losses.
        //        float ret = game.Returns()[0];
        //        if ( ret > 0 ) return WinScore + depth;
        //        if ( ret < 0 ) return -WinScore - depth;
        //        return 0;
        //    }

        //    if ( depth <= 0 ) {
        //        return eval( game );
        //    }

        //    int[] actions = game.LegalActions();
        //    if ( actions.Length == 0 ) {
        //        return eval( game );
        //    }

        //    if ( game.CurrentPlayer() == 0 ) {
        //        int value = int.MinValue;
        //        foreach ( int action in actions ) {
        //            value = Math.Max( value, DoSearch( Child( game, action ), depth - 1, alpha, beta, eval ) );
        //            if ( value >= beta ) {
        //                break;
        //            }
        //            alpha = Math.Max( alpha, value );
        //        }
        //        return value;
        //    } else {
        //        int value = int.MaxValue;
        //        foreach ( int action in actions ) {
        //            value = Math.Min( value, DoSearch( Child( game, action ), depth - 1, alpha, beta, eval ) );
        //            if ( value <= alpha ) {
        //                break;
        //            }
        //            beta = Math.Min( beta, value );
        //        }
        //        return value;
        //    }
        //}

        //private static ArcadeGame Child( ArcadeGame game, int action ) {
        //    ArcadeGame child = game.Clone();
        //    child.Apply( action );
        //    return child;
        //}
    }
}
