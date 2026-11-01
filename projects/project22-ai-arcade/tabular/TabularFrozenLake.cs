using arcade;
using System;
using System.Collections.Generic;
using System.Text;

namespace arcade {
    public class FrozenLakeTabularModelBased : FrozenLake, TabularModelBasedInterface {
        public int NumStates => Rows * Cols;

        private char TileAt( int state ) => map[state / Cols][state % Cols];
        public bool IsTerminal( int state ) => TileAt( state ) == 'H' || TileAt( state ) == 'G';
        public int[] LegalActions( int state ) => IsTerminal( state ) ? Array.Empty<int>() : new[] { 0, 1, 2, 3 };

        public MdpTransition[] Transitions( int state, int action ) {
            if ( IsTerminal( state ) || action < 0 || action >= 4 )
                throw new ArgumentException( $"Illegal move {action} from state {state}" );

            int[] dirs = slippery ? new[] { action, (action + 1) % 4, (action + 3) % 4 } : new[] { action };
            float prob = 1f / dirs.Length;

            MdpTransition[] transitions = new MdpTransition[dirs.Length];
            for ( int i = 0; i < dirs.Length; i++ ) {
                int dir = dirs[i];
                var (r, c) = Move( state / Cols, state % Cols, dir );
                int next = r * Cols + c;
                float reward = TileAt( next ) == 'G' ? 1f : 0f;
                MdpTransition transition = new MdpTransition() { Prob = prob, NextState = next, Reward = reward };
                transitions[i] = transition;
            }
            return transitions;
        }
    }

    public class FrozenLakeTabularModelFree : FrozenLake, TabularModelFreeInterface {
        public int NumStates => throw new NotImplementedException();

        private char TileAt( int state ) => map[state / Cols][state % Cols];
        private bool IsTerminal( int state ) => TileAt( state ) == 'H' || TileAt( state ) == 'G';

        public int[] LegalActions( int state ) {
            return IsTerminal( state ) ? Array.Empty<int>() : new[] { 0, 1, 2, 3 };
        }

        public void NewGame() {
            throw new NotImplementedException();
        }

        public void RandomGame() {
            throw new NotImplementedException();
        }

        public float Step( int a ) {
            throw new NotImplementedException();
        }

        int TabularModelFreeInterface.State() {
            return this.State;
        }
    }

}
