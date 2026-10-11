
namespace rl
{
    class Program
    {

        public static Random random = new Random();

        public static double PHeads()
        {
            return 0.4;
            //return random.NextDouble();
        }

        public static double ExpectedValue(double[] V, int capital, int wager)
        {
            int sHeads = capital + wager * 2;
            double heads = PHeads() * V[sHeads];
            int sTails = capital;
            double tails = (1 - PHeads()) * V[sTails];
            return heads + tails;
        }

        public static void ValueIteration()
        {
            const double theta = 1e-12;
            double[] V = new double[101];
            V[0] = 0;
            V[100] = 1;

            double delta = 0;
            do
            {
                delta = 0;
                for (int s = 1; s < 100; s++)
                {
                    double v = V[s];
                    double max = 0.0;
                    int bound = Math.Min(s, 100 - s);
                    for (int a = 1; a <= bound; a++)
                    {
                        max = Math.Max(max, ExpectedValue(V, s - a, a));
                    }
                    V[s] = max;
                    delta = Math.Max(delta, Math.Abs(v - V[s]));
                }
            } while (delta >= theta);

            int[] policy = new int[101];
            for (int s = 1; s < 100; s++)
            {
                double v = V[s];
                double max = 0.0;
                int amax = 0;
                int bound = Math.Min(s, 100 - s);
                for (int a = 1; a <= bound; a++)
                {
                    double e = ExpectedValue(V, s - a, a);
                    if (e > max)
                    {
                        max = e;
                        amax = a;
                    }
                }
                policy[s] = amax;
            }

            PrintV(V);
            for ( int i = 1; i <= 50; i++)
            {
                Console.Write($"{i:00}|");
            }
            Console.WriteLine();
            for ( int i = 1; i <= 50; i++)
            {
                Console.Write($"{policy[i]:00}|");
            }
        }

        public static void Main()
        {
            ValueIteration();
        }

        private static void PrintV(double[] V)
        {
            for (int i = 0; i < V.Length; i++)
            {
                Console.Write($"{V[i]:F2}, ");
            }
            Console.WriteLine();
        }
    }

}
