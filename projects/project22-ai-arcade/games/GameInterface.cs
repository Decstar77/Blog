using Raylib_cs;

namespace arcade {
    // Players are numbered 0..NumPlayers-1. Actions are ints in 0..NumActions-1, and
    // LegalActions() returns the subset playable from the current state.
    // Returns() gives one score per player; it's only meaningful once IsTerminal() is true.
    // ToString() renders the state as text; Render() draws it with raylib inside the given area.
    public interface ArcadeGame
    {
        public int          NumPlayers { get; }
        public int          NumActions { get; }

        public int          CurrentPlayer();
        public int[]        LegalActions();
        public void         Apply(int action);
        public bool         IsTerminal();
        public float[]      Returns();

        // Deep copy, so search algorithms (minimax, MCTS) can try moves without touching the real game.
        public ArcadeGame   Clone();

        // Draws the current state, scaled to fit inside area. Must be called between BeginDrawing/EndDrawing.
        public void         Render(Rectangle area);
    }
}
