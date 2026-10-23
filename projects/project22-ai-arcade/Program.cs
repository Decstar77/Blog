// Arcade viewer: shows each game being played so the renderers and agents can be checked.
// In the two-player games player 0 is an alpha-beta agent and player 1 plays randomly.
// Keys: 1-4 or Tab pick a game, Space pauses, N steps once while paused, R resets, Up/Down change speed.

using System.Numerics;
using Raylib_cs;
using arcade;

// Agent picks player 0's move; null means every player moves randomly.
var games = new (string Name, Func<ArcadeGame> Make, Func<ArcadeGame, int>? Agent)[]
{
    // Small enough to search to the end, so the heuristic is never used.
    ("Tic-Tac-Toe",     () => new TicTacToe(),   g => AlphaBeta.Search(g, 9, _ => 0)),
    ("Connect Four",    () => new ConnectFour(), g => AlphaBeta.Search(g, 6, s => ((ConnectFour)s).Evaluate())),
    ("Frozen Lake 4x4", () => new FrozenLake(),  null),
    ("Frozen Lake 8x8", () => new FrozenLake(FrozenLake.Map8x8, maxSteps: 200), null),
};

const int HeaderHeight = 64;
const int FooterHeight = 36;

int selected = 0;
ArcadeGame game = games[selected].Make();
var rng = new Random();
bool paused = false;
float stepDelay = 0.4f;
float timer = 0;

Raylib.SetConfigFlags(ConfigFlags.Msaa4xHint | ConfigFlags.ResizableWindow | ConfigFlags.VSyncHint);
Raylib.InitWindow(900, 760, "AI Arcade");
Raylib.SetWindowMinSize(400, 400);
Raylib.SetTargetFPS(60);

while (!Raylib.WindowShouldClose())
{
    for (int i = 0; i < games.Length; i++)
        if (Raylib.IsKeyPressed(KeyboardKey.One + i)) Select(i);
    if (Raylib.IsKeyPressed(KeyboardKey.Tab)) Select((selected + 1) % games.Length);
    if (Raylib.IsKeyPressed(KeyboardKey.R)) Select(selected);
    if (Raylib.IsKeyPressed(KeyboardKey.Space)) paused = !paused;
    if (Raylib.IsKeyPressed(KeyboardKey.Up)) stepDelay = MathF.Max(0.02f, stepDelay / 1.5f);
    if (Raylib.IsKeyPressed(KeyboardKey.Down)) stepDelay = MathF.Min(2f, stepDelay * 1.5f);

    timer += Raylib.GetFrameTime();
    bool step = paused ? Raylib.IsKeyPressed(KeyboardKey.N) : timer >= stepDelay;
    if (step)
    {
        timer = 0;
        if (!game.IsTerminal())
        {
            var agent = games[selected].Agent;
            if (agent != null && game.CurrentPlayer() == 0)
            {
                game.Apply(agent(game));
            }
            else
            {
                int[] legal = game.LegalActions();
                game.Apply(legal[rng.Next(legal.Length)]);
            }
            // Linger on the final position for a few steps before starting over.
            if (game.IsTerminal()) timer = -stepDelay * 4;
        }
        else if (!paused)
        {
            game = games[selected].Make();
        }
    }

    int w = Raylib.GetScreenWidth(), h = Raylib.GetScreenHeight();
    Raylib.BeginDrawing();
    Raylib.ClearBackground(Style.Background);

    DrawHeader(w);
    game.Render(new Rectangle(0, HeaderHeight, w, h - HeaderHeight - FooterHeight));
    Raylib.DrawText("1-4/Tab game   Space pause   N step   R reset   Up/Down speed",
                    16, h - FooterHeight + 10, 16, Style.TextDim);

    Raylib.EndDrawing();
}

Raylib.CloseWindow();

void Select(int i)
{
    selected = i;
    game = games[i].Make();
    timer = 0;
}

void DrawHeader(int w)
{
    Raylib.DrawRectangle(0, 0, w, HeaderHeight, Style.Panel);
    Raylib.DrawLine(0, HeaderHeight, w, HeaderHeight, Style.PanelEdge);
    Raylib.DrawText(games[selected].Name, 20, 18, 28, Style.Text);

    string status;
    Color color;
    if (game.IsTerminal())
    {
        float[] ret = game.Returns();
        if (game.NumPlayers == 1)
            (status, color) = ret[0] > 0 ? ("Reached the goal", Style.Highlight) : ("Episode over", Style.TextDim);
        else if (ret[0] == ret[1])
            (status, color) = ("Draw", Style.TextDim);
        else
        {
            int p = ret[0] > ret[1] ? 0 : 1;
            (status, color) = ($"Player {p + 1} wins", Style.PlayerColor(p));
        }
    }
    else if (game.NumPlayers == 1)
        (status, color) = ("Playing", Style.TextDim);
    else
        (status, color) = ($"Player {game.CurrentPlayer() + 1} to move", Style.PlayerColor(game.CurrentPlayer()));

    if (paused) status += "  (paused)";
    int tw = Raylib.MeasureText(status, 22);
    Raylib.DrawCircleV(new Vector2(w - tw - 40, HeaderHeight / 2f), 7, color);
    Raylib.DrawText(status, w - tw - 20, 21, 22, Style.Text);
}
