namespace arcade {
    public record MdpTransition {
        public float    Prob;
        public int      NextState;
        public float    Reward;
    }

    public interface TabularModelBasedInterface {
        public int              NumStates { get; }
        public int              NumActions { get; }
        public bool             IsTerminal(int state);
        public int[]            LegalActions(int state);
        public MdpTransition[]  Transitions(int state, int action);
    }

    public interface TabularModelFreeInterface {
        public int              NumStates { get; }
        public int              NumActions { get; }
        public void             NewGame();
        public void             RandomGame();
        public int              State();
        public float            Step( int a );
        public bool             IsTerminal();
        public int[]            LegalActions( int state );
    }

}
