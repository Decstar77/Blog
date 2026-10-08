using Raylib_cs;

class Program
{
    public static void Main()
    {
        // Initialize the window (Width, Height, Title)
        Raylib.InitWindow(800, 480, "Hello Raylib-cs");
        Raylib.SetTargetFPS(60);

        // Main game loop
        while (!Raylib.WindowShouldClose())
        {
            // Drawing phase
            Raylib.BeginDrawing();
            Raylib.ClearBackground(Color.RayWhite);

            Raylib.DrawText("Congrats! Raylib is working in C#!", 190, 200, 20, Color.LightGray);

            Raylib.EndDrawing();
        }

        // Close window and clean up resources
        Raylib.CloseWindow();
    }
}
