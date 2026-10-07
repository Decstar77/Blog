

namespace rl
{
    class Program
    {
        public static double ExpectedReturn(double[,] V, int ax, int ay)
        {
            double expected = 0;
            // sum_s'r P(s', r | s, a) * [r + y * V(s')]
            
        }

        public static void PolicyIteration()
        {
            double[,] V = new double[3, 3];
            int[,] policy = new int[3, 3];

            while (true)
            {
                double delta = 0;
                do
                {
                    delta = 0.0;
                    for (int x = 0; x < 3; x++)
                    {
                        for (int y = 0; y < 3; y++)
                        {
                            double v = V[x, y];
                            V[x, y] = ExpectedReturn(V);
                        }
                    }
                } while (delta > 0.01);
            }
        }

        public static void Main()
        {

        }
    }
}

