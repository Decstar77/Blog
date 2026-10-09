using Raylib_cs;

namespace game
{
    public class PlayerAction
    {

    };

    public class ThreadCoordinator
    {
        public List<Thread> Threads = new List<Thread>();
        public void WaitForAll()
        {

        }

        
    };

    public struct ThreadContext
    {
        public readonly ThreadCoordinator   Coordinator;
        public readonly int                 ThreadIndex;
    }

    public static class Simulation
    {
        public static void Update(ThreadCoordinator coordinator)
        {


            coordinator.WaitForAll();

            coordinator.StartLoop();
            while (coordinator.Looping() && coordinator.GetIndex())
            {
                
            }

        }
    }

    class Program
    {
        public static void Main()
        {
            ThreadCoordinator coordinator;

            List<Thread> threads = new List<Thread>();
            for (int i = 0; i < 8; i++)
            {
                threads.Add(new Thread(() =>  ))
            }


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
}