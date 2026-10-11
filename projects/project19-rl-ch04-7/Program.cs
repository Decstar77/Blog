namespace rl
{
    class Program
    {
        private static double Reward_CarRental = 10;
        private static double Reward_CarTransfer = -2;
        private static double MaxCars = 20;
        private static double MaxMoveable = 5;
        private static double DiscountRate = 0.9;

        private static double ProbabiltyOfCars(int n, int lamda)
        {
            double neum = Math.Pow(lamda, n);
            double de = CalculateFactorial((int)n);
            double e = Math.Exp(-lamda);
            return (neum / de) * e;
        }

        private static List<(int, double)> GetOutcomes(int location, int[] Means)
        {
            List<(int, double)> outcomes = new List<(int, double)>();
            for (int i = 0; i < 20; i++)
            {
                outcomes.Add((i, ProbabiltyOfCars(i, Means[location - 1])));
            }

            double remainingProbability = 1.0 - outcomes.Sum(outcome => outcome.Item2);
            outcomes.Add((20, remainingProbability));

            return outcomes;
        }

        private static List<(int, double)>? CachedRequestOutcomes1 = null;
        private static List<(int, double)>? CachedRequestOutcomes2 = null;
        private static List<(int, double)>? CachedReturnOutcomes1 = null;
        private static List<(int, double)>? CachedReturnOutcomes2 = null;

        private static double ExpectedReturn(double[,] V, int l1State, int l2State, int carsMoved)
        {
            int cars1 = l1State - carsMoved;
            int cars2 = l2State + carsMoved;
            double moveCost = carsMoved > 0 ? 2 * Math.Max(0, carsMoved - 1) : 2 * Math.Abs(carsMoved);
            double expectedReturn = -moveCost;

            if (cars1 > 10) expectedReturn -= 4;
            if (cars2 > 10) expectedReturn -= 4;

            int[] RequestsMeans = { 3, 4 };
            int[] ReturnMeans = { 3, 2 };

            var requestOutcomes1 = CachedRequestOutcomes1 == null ? GetOutcomes(1, RequestsMeans) : CachedRequestOutcomes1;
            var requestOutcomes2 = CachedRequestOutcomes2 == null ? GetOutcomes(2, RequestsMeans) : CachedRequestOutcomes2;
            var returnOutcomes1 = CachedReturnOutcomes1 == null ? GetOutcomes(1, ReturnMeans) : CachedReturnOutcomes1;
            var returnOutcomes2 = CachedReturnOutcomes2 == null ? GetOutcomes(2, ReturnMeans) : CachedReturnOutcomes2;

            CachedRequestOutcomes1 = requestOutcomes1;
            CachedRequestOutcomes2 = requestOutcomes2;
            CachedReturnOutcomes1 = returnOutcomes1;
            CachedReturnOutcomes2 = returnOutcomes2;

            foreach (var (requests1, pRequests1) in requestOutcomes1)
                foreach (var (requests2, pRequests2) in requestOutcomes2)
                    foreach (var (returns1, pReturns1) in returnOutcomes1)
                        foreach (var (returns2, pReturns2) in returnOutcomes2)
                        {
                            int rented1 = Math.Min(cars1, requests1);
                            int rented2 = Math.Min(cars2, requests2);
                            int next1 = Math.Min(20, cars1 - rented1 + returns1);
                            int next2 = Math.Min(20, cars2 - rented2 + returns2);

                            double probability = pRequests1 * pRequests2 * pReturns1 * pReturns2;
                            expectedReturn += probability * (Reward_CarRental * (rented1 + rented2) + 0.9 * V[next1, next2]);
                        }

            return expectedReturn;
        }

        private static bool IsMoveFeasible(int cars1, int cars2, int carsMoved)
        {
            if (cars1 - carsMoved < 0) return false;
            if (cars1 - carsMoved >= 21) return false;
            if (cars2 + carsMoved < 0) return false;
            if (cars2 + carsMoved >= 21) return false;
            return true;
        }

        private static void PolicyIteration()
        {
            // Step 1. Initialization
            Console.WriteLine("Begining Policy Iteration");
            double[,] V = new double[21, 21];
            int[,] policy = new int[21, 21];

            while (true)
            {
                // Step 2. Policy Evaluation
                Console.WriteLine("Begining Policy Evaluation");
                double delta = 1.0;
                double theta = 0.01;
                while (delta >= theta)
                {
                    delta = 0;
                    for (int cars1 = 0; cars1 < 21; cars1++)
                    {
                        for (int cars2 = 0; cars2 < 21; cars2++)
                        {
                            double v = V[cars1, cars2];
                            V[cars1, cars2] = ExpectedReturn(V, cars1, cars2, policy[cars1, cars2]);
                            delta = Math.Max(delta, Math.Abs(v - V[cars1, cars2]));
                        }
                    }
                }

                // Step 3. Policy Improvement
                Console.WriteLine("Begining Policy Improvement");
                bool stable = true;
                for (int cars1 = 0; cars1 < 21; cars1++)
                {
                    for (int cars2 = 0; cars2 < 21; cars2++)
                    {
                        int oldAction = policy[cars1, cars2];
                        int bestAction = 0;
                        double bestValue = double.MinValue;
                        for (int carsMoved = -5; carsMoved <= 5; carsMoved++)
                        {
                            if (IsMoveFeasible(cars1, cars2, carsMoved) == false) continue;

                            double v = ExpectedReturn(V, cars1, cars2, carsMoved);
                            if (v > bestValue)
                            {
                                bestValue = v;
                                bestAction = carsMoved;
                            }
                        }
                        policy[cars1, cars2] = bestAction;
                        if (oldAction != bestAction)
                        {
                            stable = false;
                        }
                    }
                }

                PrintPolicy(policy);
                if (stable) break;
            }
        }

        public static void Main()
        {
            PolicyIteration();
        }

        private static void PrintPolicy(int[,] policy)
        {
            for (int i = 0; i < policy.GetLength(0); i++)
            {
                Console.Write("|");
                for (int j = 0; j < policy.GetLength(1); j++)
                {
                    Console.Write(policy[i, j]);
                    Console.Write("|");
                }
                Console.WriteLine();
            }
        }

        private static long CalculateFactorial(int n)
        {
            if (n < 0) throw new ArgumentException("Factorial is not defined for negative numbers.");

            long factorial = 1;
            for (int i = 1; i <= n; i++)
            {
                factorial *= i;
            }
            return factorial;
        }
    }
}



