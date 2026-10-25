namespace arcade {
    public record MdpTransition {
        public float    Prob;
        public int      NextState;
        public float    Reward;
    }

    public interface DPInterface {
        public int              NumStates { get; }
        public int              NumActions { get; }
        public bool             IsTerminal(int state);
        public int[]            LegalActions(int state);
        public MdpTransition[]  Transitions(int state, int action);
    }

    public interface MonteCarloInterface {
        public int      NumStates => 0;
        public int      NumActions => 0;
        public void     NewGame();
        public void     RandomGame();
        public int      State();
        public float    Step( int a );
        public bool     IsTerminal();
        public int[]    LegalActions( int state );
    }

    public class FrozenLakeTabular : FrozenLake, DPInterface {
        public int      NumStates => Rows * Cols;

        private char    TileAt( int state ) => map[state / Cols][state % Cols];
        public bool     IsTerminal( int state ) => TileAt( state ) == 'H' || TileAt( state ) == 'G';
        public int[]    LegalActions( int state ) => IsTerminal( state ) ? Array.Empty<int>() : new[] { 0, 1, 2, 3 };

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
}
