
using System.Xml;

namespace rl
{
    class Program
    {
        public const int Left = 0;
        public const int Right = 1;
        public const int Up = 2;
        public const int Down = 3;
        public const int UpLeft = 4;
        public const int UpRight = 5;
        public const int DownLeft = 6;
        public const int DownRight = 7;
        public const int NoMove = 8;

        public const int CellWidth = 10;
        public const int CellHeight = 7;

        public static Random random = new Random();

        public record State
        {
            public int x;
            public int y;
        }

        public static int ChooseNextAction(State state, double[,,] Q, int ActionCount)
        {
            const double Epsilon = 0.1;
            if (random.NextDouble() > Epsilon)
            {
                int maxAction = 0;
                double max = double.MinValue;
                for (int action = 0; action < ActionCount; action++)
                {
                    if (Q[state.x, state.y, action] > max)
                    {
                        max = Q[state.x, state.y, action];
                        maxAction = action;
                    }
                }
                return maxAction;
            }
            else
            {
                return random.Next(ActionCount);
            }
        }

        public static State TakeNextAction(State state, int action, int[,] Wind)
        {
            int dx = 0;
            int dy = 0;
            switch (action)
            {
                case Left: dx = -1; dy = 0; break;
                case Right: dx = 1; dy = 0; break;
                case Up: dx = 0; dy = 1; break;
                case Down: dx = 0; dy = -1; break;
                case UpLeft: dx = -1; dy = 1; break;
                case UpRight: dx = 1; dy = 1; break;
                case DownLeft: dx = -1; dy = -1; break;
                case DownRight: dx = 1; dy = -1; break;
                case NoMove: dx = 0; dy = 0; break;
                default: throw new ArgumentException("Invalid action.");
            }

            // Apply wind
            dy -= Wind[state.x, state.y];

            State nextState = new State();
            nextState.x = Math.Clamp(state.x + dx, 0, CellWidth - 1);
            nextState.y = Math.Clamp(state.y + dy, 0, CellHeight - 1);
            return nextState;
        }

        public static State TakeNextActionStochastic(State state, int action, int[,] Wind)
        {
            int dx = 0;
            int dy = 0;
            switch (action)
            {
                case Left: dx = -1; dy = 0; break;
                case Right: dx = 1; dy = 0; break;
                case Up: dx = 0; dy = 1; break;
                case Down: dx = 0; dy = -1; break;
                case UpLeft: dx = -1; dy = 1; break;
                case UpRight: dx = 1; dy = 1; break;
                case DownLeft: dx = -1; dy = -1; break;
                case DownRight: dx = 1; dy = -1; break;
                case NoMove: dx = 0; dy = 0; break;
                default: throw new ArgumentException("Invalid action.");
            }

            int wind = Wind[state.x, state.y];
            if (wind != 0)
            {
                // Stochastic wind: mean, mean + 1, or mean - 1, each with probability 1/3
                wind += random.Next(3) - 1;
            }
            dy -= wind;

            State nextState = new State();
            nextState.x = Math.Clamp(state.x + dx, 0, CellWidth - 1);
            nextState.y = Math.Clamp(state.y + dy, 0, CellHeight - 1);
            return nextState;
        }

        public static void Exericse6_9(int episodeCount, int actionCount)
        {
            int[,] Wind = new int[CellWidth, CellHeight];
            for (int x = 3; x < 9; x++)
            {
                for (int y = 0; y < CellHeight; y++)
                {
                    Wind[x, y] = (x == 6 || x == 7) ? 2 : 1;
                }
            }

            // PrintWind(Wind);

            const double Alpha = 0.5;
            double[,,] Q = new double[CellWidth, CellHeight, actionCount];
            State terminalState = new State() { x = 7, y = CellHeight / 2 };
            State startState = new State() { x = 0, y = CellHeight / 2 };
            for (int episodeIndex = 0; episodeIndex < episodeCount; episodeIndex++)
            {
                State currentState = startState;
                int action = ChooseNextAction(currentState, Q, actionCount);
                while (currentState != terminalState)
                {
                    State nextState = TakeNextAction(currentState, action, Wind);
                    double reward = nextState == terminalState ? 0 : -1;
                    int nextAction = ChooseNextAction(nextState, Q, actionCount);

                    Q[currentState.x, currentState.y, action] += Alpha * (reward + Q[nextState.x, nextState.y, nextAction] - Q[currentState.x, currentState.y, action]);
                    currentState = nextState;
                    action = nextAction;
                }
            }

            PrintShortestPath(Q, startState, terminalState, Wind, actionCount);
        }

        public static void Exercise6_10(int episodeCount)
        {
            int[,] Wind = new int[CellWidth, CellHeight];
            for (int x = 3; x < 9; x++)
            {
                for (int y = 0; y < CellHeight; y++)
                {
                    Wind[x, y] = (x == 6 || x == 7) ? 2 : 1;
                }
            }

            // PrintWind(Wind);
            const int ActionCount = 8;
            const double Alpha = 0.5;
            double[,,] Q = new double[CellWidth, CellHeight, ActionCount];
            State terminalState = new State() { x = 7, y = CellHeight / 2 };
            State startState = new State() { x = 0, y = CellHeight / 2 };
            for (int episodeIndex = 0; episodeIndex < episodeCount; episodeIndex++)
            {
                State currentState = startState;
                int action = ChooseNextAction(currentState, Q, ActionCount);
                while (currentState != terminalState)
                {
                    State nextState = TakeNextActionStochastic(currentState, action, Wind);
                    double reward = nextState == terminalState ? 0 : -1;
                    int nextAction = ChooseNextAction(nextState, Q, ActionCount);

                    Q[currentState.x, currentState.y, action] += Alpha * (reward + Q[nextState.x, nextState.y, nextAction] - Q[currentState.x, currentState.y, action]);
                    currentState = nextState;
                    action = nextAction;
                }
            }

            PrintAverageGreedySteps(Q, startState, terminalState, Wind, ActionCount, 10_000);
        }

        public static void PrintAverageGreedySteps(double[,,] Q, State start, State end, int[,] Wind, int actionCount, int runs)
        {
            const int MaxSteps = 1_000;
            long totalSteps = 0;
            int failures = 0;
            for (int run = 0; run < runs; run++)
            {
                State current = start;
                int steps = 0;
                while (current != end && steps < MaxSteps)
                {
                    int bestAction = 0;
                    double best = double.MinValue;
                    for (int action = 0; action < actionCount; action++)
                    {
                        if (Q[current.x, current.y, action] > best)
                        {
                            best = Q[current.x, current.y, action];
                            bestAction = action;
                        }
                    }
                    current = TakeNextActionStochastic(current, bestAction, Wind);
                    steps++;
                }

                if (current == end) totalSteps += steps;
                else failures++;
            }

            int successes = runs - failures;
            Console.WriteLine(successes > 0
                ? $"Greedy policy under stochastic wind: average {(double)totalSteps / successes:F2} steps over {successes} runs ({failures} hit the {MaxSteps}-step cap)."
                : "Greedy policy never reached the goal.");
        }

        public static void Main()
        {
            Exericse6_9(50_000, 4);
            Console.WriteLine("==============================");
            Exericse6_9(50_000, 8);
            Console.WriteLine("==============================");
            Exericse6_9(50_000, 9);
            // Exercise6_10(50_000);
        }

        public static void PrintShortestPath(double[,,] Q, State start, State end, int[,] Wind, int actionCount)
        {
            int width = Q.GetLength(0);
            int height = Q.GetLength(1);
            int maxSteps = width * height * 4;

            var path = new List<State> { start };
            var visited = new HashSet<State> { start };
            State current = start;
            bool reachedGoal = false;

            while (path.Count <= maxSteps)
            {
                if (current == end)
                {
                    reachedGoal = true;
                    break;
                }

                int bestAction = 0;
                double best = double.MinValue;
                for (int action = 0; action < actionCount; action++)
                {
                    if (Q[current.x, current.y, action] > best)
                    {
                        best = Q[current.x, current.y, action];
                        bestAction = action;
                    }
                }

                current = TakeNextAction(current, bestAction, Wind);
                path.Add(current);
                if (!visited.Add(current) && current != end)
                {
                    break; // greedy policy is looping
                }
            }

            var stepAt = new Dictionary<State, int>();
            for (int i = 0; i < path.Count; i++)
            {
                stepAt.TryAdd(path[i], i);
            }

            Console.WriteLine();
            for (int y = 0; y < height; y++)
            {
                Console.Write("|");
                for (int x = 0; x < width; x++)
                {
                    var cell = new State { x = x, y = y };
                    string label;
                    if (cell == start) label = " S";
                    else if (cell == end) label = " G";
                    else if (stepAt.TryGetValue(cell, out int step)) label = $"{step,2}";
                    else label = " .";
                    Console.Write($"{label}|");
                }
                Console.WriteLine();
            }

            Console.WriteLine(reachedGoal
                ? $"Greedy path reaches goal in {path.Count - 1} steps."
                : $"Greedy path did not reach goal (stopped after {path.Count - 1} steps, loop detected or step limit hit).");
        }

        private static void PrintQ_Greedy(double[,,] Q, int actionCount)
        {
            for (int y = 0; y < Q.GetLength(1); y++)
            {
                Console.Write("|");
                for (int x = 0; x < Q.GetLength(0); x++)
                {
                    int maxAction = 0;
                    double max = double.MinValue;
                    for (int action = 0; action < actionCount; action++)
                    {
                        if (Q[x, y, action] > max)
                        {
                            max = Q[x, y, action];
                            maxAction = action;
                        }
                    }

                    string asci = "";
                    if (maxAction == 0)
                    {
                        asci = "←";
                    }
                    else if (maxAction == 1)
                    {
                        asci = "→";
                    }
                    else if (maxAction == 2)
                    {
                        asci = "↓";
                    }
                    else
                    {
                        asci = "↑";
                    }

                    Console.Write($"{asci}|");
                }
                Console.WriteLine();
            }
        }

        private static void PrintWind(int[,] Wind)
        {
            for (int y = 0; y < Wind.GetLength(1); y++)
            {
                Console.Write("|");
                for (int x = 0; x < Wind.GetLength(0); x++)
                {
                    Console.Write($"{Wind[x, y]}|");
                }
                Console.WriteLine();
            }
        }
    }

}