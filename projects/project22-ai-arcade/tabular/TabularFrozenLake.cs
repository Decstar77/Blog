using arcade;
using System;
using System.Collections.Generic;
using System.Text;

namespace arcade {
    public class FrozenLakeTabularModelBased : FrozenLake, TabularModelBasedInterface {
        public int NumStates => Rows * Cols;

        public FrozenLakeTabularModelBased( string[]? map = null, bool slippery = true )
            : base( map, slippery ) {
        }

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
        private readonly int startState;

        public int NumStates => Rows * Cols;

        public FrozenLakeTabularModelFree( string[]? map = null, bool slippery = true, int maxSteps = 100, int? seed = null )
            : base( map, slippery, maxSteps, seed ) {
            startState = State;
        }

        private char TileAt( int state ) => map[state / Cols][state % Cols];
        private bool IsTerminal( int state ) => TileAt( state ) == 'H' || TileAt( state ) == 'G';

        public int[] LegalActions( int state ) {
            return IsTerminal( state ) ? Array.Empty<int>() : new[] { 0, 1, 2, 3 };
        }

        public void NewGame() {
            Reset( startState );
        }

        public void RandomGame() {
            int state;
            do {
                state = rng.Next( NumStates );
            } while ( IsTerminal( state ) );
            Reset( state );
        }

        public float Step( int a ) {
            Apply( a );
            return TileAt( State ) == 'G' ? 1f : 0f;
        }

        int TabularModelFreeInterface.State() {
            return this.State;
        }

        private void Reset( int state ) {
            row = state / Cols;
            col = state % Cols;
            steps = 0;
            lastAction = -1;
            lastDir = -1;
        }
    }

}
