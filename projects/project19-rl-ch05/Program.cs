namespace rl
{
    class Program
    {
        public static BlackJack blackjack = new BlackJack();
        public static Random random = new Random();

        // State: (player sum, dealer showing card 1..10, usable ace)
        record struct State(int PlayerSum, int DealerShowing, bool UsableAce);

        public static void MCFirstVisit(int episodeCount) // Chapter 5.1
        {
            // Indexed [playerSum 0..21, dealerShowing 0..10, usableAce 0/1]
            double[,,] V = new double[22, 11, 2];
            int[,,] N = new int[22, 11, 2];

            for (int episodeIndex = 0; episodeIndex < episodeCount; episodeIndex++)
            {
                // Each step is (state before the action, reward received after it)
                var episode = new List<(State S, int R)>();

                GameResult result = blackjack.NewGame();
                if (result != GameResult.Playing)
                {
                    // Natural: the game ends immediately from the initial state
                    episode.Add((CurrentState(), RewardForGameResult(result)));
                }

                while (result == GameResult.Playing)
                {
                    State s = CurrentState();
                    int action = s.PlayerSum < 20 ? 1 : 0; // hit below 20, stick on 20 or 21
                    result = blackjack.Play(action);
                    episode.Add((s, RewardForGameResult(result)));
                }

                int G = 0; // gamma = 1
                for (int t = episode.Count - 1; t >= 0; t--)
                {
                    G += episode[t].R;
                    State s = episode[t].S;

                    bool seenEarlier = false;
                    for (int k = 0; k < t; k++)
                    {
                        if (episode[k].S == s) { seenEarlier = true; break; }
                    }
                    if (seenEarlier) continue;

                    int a = s.UsableAce ? 1 : 0;
                    N[s.PlayerSum, s.DealerShowing, a]++;
                    V[s.PlayerSum, s.DealerShowing, a] += (G - V[s.PlayerSum, s.DealerShowing, a]) / N[s.PlayerSum, s.DealerShowing, a];
                }
            }

            Console.WriteLine($"=== First-visit MC, {episodeCount:N0} episodes ===");
            PrintV(V, usableAce: true);
            PrintV(V, usableAce: false);
        }

        public static void MCExploringStarts(int episodeCount)
        {
            double[,,,] Q = new double[22, 11, 2, 2];
            int[,,,] N = new int[22, 11, 2, 2];
            int[,,] Policy = new int[22, 11, 2];

            // Soft init the policy, hit on every value other than 20 or 21
            for (int i = 0; i < 20; i++)
                for (int j = 0; j < 11; j++)
                    for (int k = 0; k < 2; k++)
                    {
                        Policy[i, j, k] = 1;
                    }


            for (int episodeIndex = 0; episodeIndex < episodeCount; episodeIndex++)
            {
                var episode = new List<(State S, int A, int R)>();

                GameResult result = blackjack.RandomGame();
                bool first = true;
                while (result == GameResult.Playing)
                {
                    State s = CurrentState();
                    int action = first ? random.Next(2) : Policy[s.PlayerSum, s.DealerShowing, s.UsableAce ? 1 : 0];
                    first = false;
                    result = blackjack.Play(action);
                    episode.Add((s, action, RewardForGameResult(result)));
                }

                int G = 0; // gamma = 1
                for (int t = episode.Count - 1; t >= 0; t--)
                {
                    G += episode[t].R;
                    State s = episode[t].S;
                    int action = episode[t].A;

                    bool seenEarlier = false;
                    for (int k = 0; k < t; k++)
                    {
                        if (episode[k].S == s && episode[k].A == action) { seenEarlier = true; break; }
                    }
                    if (seenEarlier) continue;

                    int ace = s.UsableAce ? 1 : 0;
                    N[s.PlayerSum, s.DealerShowing, ace, action]++;
                    Q[s.PlayerSum, s.DealerShowing, ace, action] += (G - Q[s.PlayerSum, s.DealerShowing, ace, action]) / N[s.PlayerSum, s.DealerShowing, ace, action];
                    Policy[s.PlayerSum, s.DealerShowing, ace] = Q[s.PlayerSum, s.DealerShowing, ace, 0] > Q[s.PlayerSum, s.DealerShowing, ace, 1] ? 0 : 1;
                }
            }

            PrintPolicy(Policy, false);
        }

        public static void MCOnPolicyEpsilonSoft(int episodeCount)
        {
            double[,,,] Q = new double[22, 11, 2, 2];
            int[,,,] N = new int[22, 11, 2, 2];
            // Because we only have 2 actions, hit or stay we can use one number and derive the other as 1 - P[...]. Addtionally this is the probablities that we stick
            double[,,] Policy = new double[22, 11, 2];
            const double Epsilon = 0.1;

            // Soft init the policy, hit on every value other than 20 or 21
            for (int i = 0; i < 22; i++)
                for (int j = 0; j < 11; j++)
                    for (int k = 0; k < 2; k++)
                        Policy[i, j, k] = 0.5;


            for (int episodeIndex = 0; episodeIndex < episodeCount; episodeIndex++)
            {
                var episode = new List<(State S, int A, int R)>();

                GameResult result = blackjack.NewGame();
                if (result != GameResult.Playing)
                {
                    continue;
                }

                while (result == GameResult.Playing)
                {
                    State s = CurrentState();
                    int action = Policy[s.PlayerSum, s.DealerShowing, s.UsableAce ? 1 : 0] > random.NextDouble() ? 0 : 1;
                    result = blackjack.Play(action);
                    episode.Add((s, action, RewardForGameResult(result)));
                }

                int G = 0;
                for (int t = episode.Count - 1; t >= 0; t--)
                {
                    G += episode[t].R;
                    State s = episode[t].S;
                    int action = episode[t].A;

                    bool seenEarlier = false;
                    for (int k = 0; k < t; k++)
                    {
                        if (episode[k].S == s && episode[k].A == action) { seenEarlier = true; break; }
                    }
                    if (seenEarlier) continue;

                    int ace = s.UsableAce ? 1 : 0;
                    N[s.PlayerSum, s.DealerShowing, ace, action]++;
                    Q[s.PlayerSum, s.DealerShowing, ace, action] += (G - Q[s.PlayerSum, s.DealerShowing, ace, action]) / N[s.PlayerSum, s.DealerShowing, ace, action];
                    int greedAction = Q[s.PlayerSum, s.DealerShowing, ace, 0] > Q[s.PlayerSum, s.DealerShowing, ace, 1] ? 0 : 1;
                    Policy[s.PlayerSum, s.DealerShowing, ace] = greedAction == 0 ? 1 - Epsilon + Epsilon / 2.0 : Epsilon / 2.0;
                }
            }

            int[,,] greedPolicy = new int[22, 11, 2];
            for (int i = 0; i < 22; i++)
                for (int j = 0; j < 11; j++)
                    for (int k = 0; k < 2; k++)
                        greedPolicy[i, j, k] = Policy[i, j, k] > 1 - Policy[i, j, k] ? 0 : 1;

            PrintPolicy(greedPolicy, true);
        }

        public static void MCOffPolicyControl(int episodeCount)
        {
            double[,,,] Q = new double[22, 11, 2, 2];
            double[,,,] C = new double[22, 11, 2, 2];

            int[,,] TargetPolicy = new int[22, 11, 2];
            double[,,] BehaviorPolicy = new double[22, 11, 2];
            for (int i = 0; i < 22; i++)
                for (int j = 0; j < 11; j++)
                    for (int k = 0; k < 2; k++)
                    {
                        BehaviorPolicy[i, j, k] = 0.5;
                    }

            for (int episodeIndex = 0; episodeIndex < episodeCount; episodeIndex++)
            {
                var episode = new List<(State S, int A, int R)>();

                GameResult result = blackjack.NewGame();
                if (result != GameResult.Playing)
                {
                    continue;
                }

                while (result == GameResult.Playing)
                {
                    State s = CurrentState();
                    int action = BehaviorPolicy[s.PlayerSum, s.DealerShowing, s.UsableAce ? 1 : 0] > random.NextDouble() ? 0 : 1;
                    result = blackjack.Play(action);
                    episode.Add((s, action, RewardForGameResult(result)));
                }

                int G = 0;
                double W = 1;
                for (int t = episode.Count - 1; t >= 0; t--)
                {
                    G += episode[t].R;
                    State s = episode[t].S;
                    int action = episode[t].A;
                    int ace = s.UsableAce ? 1 : 0;

                    C[s.PlayerSum, s.DealerShowing, ace, action] += W;
                    Q[s.PlayerSum, s.DealerShowing, ace, action] += (W / C[s.PlayerSum, s.DealerShowing, ace, action]) * (G - Q[s.PlayerSum, s.DealerShowing, ace, action]);
                    TargetPolicy[s.PlayerSum, s.DealerShowing, ace] = Q[s.PlayerSum, s.DealerShowing, ace, 0] > Q[s.PlayerSum, s.DealerShowing, ace, 1] ? 0 : 1;
                    if (TargetPolicy[s.PlayerSum, s.DealerShowing, ace] != action)
                    {
                        break;
                    }
                    W = W * (1.0 / BehaviorPolicy[s.PlayerSum, s.DealerShowing, ace]);
                }
            }
            
            PrintPolicy(TargetPolicy, true);
        }

        public static void Main()
        {
            //MCFirstVisit(10_000);
            //MCFirstVisit(500_000);
            //MCExploringStarts(10_000);
            //MCExploringStarts(500_000);
            //MCOnPolicyEpsilonSoft(10_000);
            //MCOnPolicyEpsilonSoft(500_000);
            //MCOffPolicyControl(10_000);
            MCOffPolicyControl(500_000);
        }

        private static State CurrentState()
        {
            return new State(blackjack.PlayerSum, blackjack.DealerShowing, blackjack.PlayerHasUsableAce);
        }

        private static int RewardForGameResult(GameResult res)
        {
            switch (res)
            {
                case GameResult.Loss: return -1;
                case GameResult.Draw: return 0;
                case GameResult.Win: return 1;
                case GameResult.Playing:
                default:
                    return 0;
            }
        }

        // Prints player sums 12..21 (rows) against dealer showing A..10 (columns), as in Figure 5.1
        private static void PrintV(double[,,] V, bool usableAce)
        {
            int a = usableAce ? 1 : 0;
            Console.WriteLine(usableAce ? "Usable ace:" : "No usable ace:");
            Console.Write("Sum |");
            for (int d = 1; d <= 10; d++)
            {
                Console.Write($"{(d == 1 ? "A" : d.ToString()),6}|");
            }
            Console.WriteLine();
            for (int p = 21; p >= 12; p--)
            {
                Console.Write($"{p,3} |");
                for (int d = 1; d <= 10; d++)
                {
                    Console.Write($"{V[p, d, a],6:F2}|");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }

        private static void PrintPolicy(int[,,] policy, bool usableAce)
        {
            int a = usableAce ? 1 : 0;
            Console.WriteLine(usableAce ? "Usable ace:" : "No usable ace:");
            Console.Write("Sum |");
            for (int d = 1; d <= 10; d++)
            {
                Console.Write($"{(d == 1 ? "A" : d.ToString()),6}|");
            }
            Console.WriteLine();
            for (int p = 21; p >= 12; p--)
            {
                Console.Write($"{p,3} |");
                for (int d = 1; d <= 10; d++)
                {
                    string v = policy[p, d, a] == 1 ? "hit" : "stick";
                    Console.Write($"{v,6}|");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
    }
}
