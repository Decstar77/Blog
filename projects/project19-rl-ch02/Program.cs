// Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed.)
// Chapter 2: Multi-armed Bandits
//
// Usage: dotnet run -- <experiment>      (no argument lists the experiments)
//
// Suggested order: Rng.ArgMax -> Bandit -> EpsilonGreedyAgent -> Simulator.Run -> fig2.2,
// then the remaining agents and experiments in section order.

using System.Globalization;

var experiments = new (string Name, string Desc, Action Run)[]
{
    ("fig2.2", "10-armed testbed: greedy vs epsilon-greedy (2.3)", Experiments.Fig2_2),
    ("ex2.5", "Nonstationary testbed: sample average vs constant step size (2.5)", Experiments.Ex2_5),
    ("fig2.3", "Optimistic initial values (2.6)", Experiments.Fig2_3),
    ("fig2.4", "UCB vs epsilon-greedy (2.7)", Experiments.Fig2_4),
    ("fig2.5", "Gradient bandit with and without baseline (2.8)", Experiments.Fig2_5),
    ("fig2.6", "Parameter study of all four algorithms (2.10)", Experiments.Fig2_6),
    ("ex2.11", "Parameter study on the nonstationary testbed (2.10)", Experiments.Ex2_11),
};

if (args.Length == 0)
{
    Console.WriteLine("Experiments:");
    foreach (var e in experiments)
        Console.WriteLine($"  {e.Name,-8} {e.Desc}");
    return;
}

var selected = experiments.FirstOrDefault(e => e.Name == args[0]);
if (selected.Run is null)
{
    Console.WriteLine($"Unknown experiment '{args[0]}'");
    return;
}
selected.Run();

static class Rng
{
    public static readonly Random Shared = new(1234);

    // Box-Muller transform
    public static double Normal(double mean = 0.0, double stdDev = 1.0)
    {
        double u1 = 1.0 - Shared.NextDouble();
        double u2 = Shared.NextDouble();
        return mean + stdDev * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    public static int ArgMax(double[] values)
    {
        // TODO() Return the index of the largest value, breaking ties uniformly at random.
        //        (Taking the first max biases the greedy agent towards arm 0 when all Q are equal.)
        throw new NotImplementedException();
    }
}

// The k-armed testbed from section 2.3.
class Bandit
{
    public int K { get; }
    public double[] QStar { get; }

    readonly double qStarMean;
    readonly bool nonstationary;

    public Bandit(int k = 10, double qStarMean = 0.0, bool nonstationary = false)
    {
        K = k;
        QStar = new double[k];
        this.qStarMean = qStarMean;
        this.nonstationary = nonstationary;
        Reset();
    }

    public void Reset()
    {
        // TODO() Stationary: draw each q*(a) from N(qStarMean, 1).
        //        Nonstationary (Exercise 2.5): start every q*(a) at the same value (qStarMean).
        throw new NotImplementedException();
    }

    public int OptimalAction()
    {
        // TODO() Return argmax_a q*(a). Used to measure "% optimal action".
        throw new NotImplementedException();
    }

    public double Step(int action)
    {
        // TODO() Return a reward drawn from N(q*(action), 1).
        //        Nonstationary (Exercise 2.5): afterwards, add an independent N(0, 0.01) increment
        //        to every q*(a) so the true values take a random walk.
        throw new NotImplementedException();
    }
}

interface IAgent
{
    void Reset();
    int SelectAction();
    void Update(int action, double reward);
}

// Action-value method with epsilon-greedy selection (sections 2.2, 2.4, 2.5, 2.6).
class EpsilonGreedyAgent : IAgent
{
    readonly int k;
    readonly double epsilon;
    readonly double? alpha; // null = sample average (step size 1/n)
    readonly double initialQ;

    readonly double[] q;
    readonly int[] n;

    public EpsilonGreedyAgent(int k, double epsilon, double? alpha = null, double initialQ = 0.0)
    {
        this.k = k;
        this.epsilon = epsilon;
        this.alpha = alpha;
        this.initialQ = initialQ;
        q = new double[k];
        n = new int[k];
        Reset();
    }

    public void Reset()
    {
        // TODO() Set every Q(a) to initialQ and every N(a) to 0.
        throw new NotImplementedException();
    }

    public int SelectAction()
    {
        // TODO() With probability epsilon pick a uniformly random action, otherwise the greedy one.
        throw new NotImplementedException();
    }

    public void Update(int action, double reward)
    {
        // TODO() Incremental update, eq. 2.3: Q <- Q + stepSize * (R - Q).
        //        stepSize is 1/N(a) for sample averages, or the constant alpha (eq. 2.5).
        // TODO() (optional, Exercise 2.7) Add a mode using the unbiased constant-step-size trick:
        //        stepSize = alpha / o_n, with o_n = o_{n-1} + alpha * (1 - o_{n-1}), o_0 = 0.
        throw new NotImplementedException();
    }
}

// Upper-confidence-bound action selection (section 2.7).
class UcbAgent : IAgent
{
    readonly int k;
    readonly double c;

    readonly double[] q;
    readonly int[] n;
    int t;

    public UcbAgent(int k, double c)
    {
        this.k = k;
        this.c = c;
        q = new double[k];
        n = new int[k];
        Reset();
    }

    public void Reset()
    {
        // TODO() Zero Q, N and the time step t.
        throw new NotImplementedException();
    }

    public int SelectAction()
    {
        // TODO() Eq. 2.10: A_t = argmax_a [ Q(a) + c * sqrt(ln t / N(a)) ].
        //        An action with N(a) = 0 counts as maximising, so try every arm once first.
        throw new NotImplementedException();
    }

    public void Update(int action, double reward)
    {
        // TODO() Sample-average update of Q(action), as in EpsilonGreedyAgent.
        throw new NotImplementedException();
    }
}

// Gradient bandit algorithm (section 2.8).
class GradientAgent : IAgent
{
    readonly int k;
    readonly double alpha;
    readonly bool useBaseline;

    readonly double[] h;  // action preferences H(a)
    readonly double[] pi; // softmax probabilities pi(a)
    double avgReward;
    int t;

    public GradientAgent(int k, double alpha, bool useBaseline = true)
    {
        this.k = k;
        this.alpha = alpha;
        this.useBaseline = useBaseline;
        h = new double[k];
        pi = new double[k];
        Reset();
    }

    public void Reset()
    {
        // TODO() Zero the preferences, the average reward baseline and t.
        throw new NotImplementedException();
    }

    public int SelectAction()
    {
        // TODO() Eq. 2.11: compute pi(a) = exp(H(a)) / sum_b exp(H(b)), then sample an action
        //        from that distribution. Subtract max(H) before exponentiating to avoid overflow.
        throw new NotImplementedException();
    }

    public void Update(int action, double reward)
    {
        // TODO() Update the baseline: the incremental average of all rewards so far
        //        (keep it at 0 when useBaseline is false).
        // TODO() Eq. 2.12: H(A)  <- H(A) + alpha * (R - baseline) * (1 - pi(A))
        //                  H(a)  <- H(a) - alpha * (R - baseline) * pi(a)      for a != A
        throw new NotImplementedException();
    }
}

// Per-step curves averaged over all runs.
record RunResult(double[] AvgReward, double[] PctOptimal);

static class Simulator
{
    public static RunResult Run(Bandit bandit, IAgent agent, int runs = 2000, int steps = 1000)
    {
        // TODO() For each run: reset the bandit and the agent, then for each step select an
        //        action, take it, and update the agent. Accumulate the reward and whether the
        //        action was optimal per step, then divide by the number of runs.
        //        Note that on the nonstationary bandit the optimal action changes over time.
        throw new NotImplementedException();
    }

    // Writes out/<name>.csv with one reward and one %-optimal column per series.
    public static void Report(string name, params (string Label, RunResult Result)[] series)
    {
        var inv = CultureInfo.InvariantCulture;
        int steps = series[0].Result.AvgReward.Length;

        Directory.CreateDirectory("out");
        string path = Path.Combine("out", name + ".csv");
        using (var w = new StreamWriter(path))
        {
            w.WriteLine("step," + string.Join(",", series.Select(s => $"{s.Label} reward,{s.Label} optimal")));
            for (int i = 0; i < steps; i++)
            {
                var cols = series.Select(s =>
                    s.Result.AvgReward[i].ToString("F4", inv) + "," + s.Result.PctOptimal[i].ToString("F4", inv));
                w.WriteLine((i + 1) + "," + string.Join(",", cols));
            }
        }

        Console.WriteLine($"{name} (final step, written to {path})");
        foreach (var (label, result) in series)
            Console.WriteLine($"  {label,-28} reward {result.AvgReward[^1].ToString("F3", inv)}   optimal {result.PctOptimal[^1].ToString("P1", inv)}");
    }
}

static class Experiments
{
    public static void Fig2_2()
    {
        // TODO() Figure 2.2: on the stationary 10-armed testbed, compare sample-average agents
        //        with epsilon = 0, 0.01 and 0.1 over 2000 runs of 1000 steps. Report all three.
        //        Then answer Exercise 2.3: which is best in the long run, and by how much?
        throw new NotImplementedException();
    }

    public static void Ex2_5()
    {
        // TODO() Exercise 2.5: on the nonstationary testbed, compare epsilon = 0.1 with sample
        //        averages against epsilon = 0.1 with constant alpha = 0.1, over 10000 steps.
        throw new NotImplementedException();
    }

    public static void Fig2_3()
    {
        // TODO() Figure 2.3: optimistic greedy (Q1 = 5, epsilon = 0) vs realistic epsilon-greedy
        //        (Q1 = 0, epsilon = 0.1), both with alpha = 0.1.
        //        Then answer Exercise 2.6: what causes the early spike in the optimistic curve?
        throw new NotImplementedException();
    }

    public static void Fig2_4()
    {
        // TODO() Figure 2.4: UCB with c = 2 vs sample-average epsilon-greedy with epsilon = 0.1.
        //        Then answer Exercise 2.8: why does UCB spike on step 11?
        throw new NotImplementedException();
    }

    public static void Fig2_5()
    {
        // TODO() Figure 2.5: gradient bandit with alpha = 0.1 and 0.4, each with and without the
        //        baseline, on a testbed whose q*(a) are drawn from N(+4, 1) instead of N(0, 1).
        throw new NotImplementedException();
    }

    public static void Fig2_6()
    {
        // TODO() Figure 2.6: for each algorithm, sweep its parameter over powers of two and print
        //        the average reward over the first 1000 steps:
        //          epsilon-greedy     epsilon = 1/128 .. 1/4
        //          gradient bandit    alpha   = 1/32  .. 4
        //          UCB                c       = 1/16  .. 4
        //          optimistic greedy  Q1      = 1/4   .. 4   (alpha = 0.1)
        throw new NotImplementedException();
    }

    public static void Ex2_11()
    {
        // TODO() Exercise 2.11: repeat the parameter study on the nonstationary testbed from
        //        Exercise 2.5, adding constant-step-size epsilon-greedy with alpha = 0.1.
        //        Use 200000-step runs and score each setting by the average reward over the
        //        last 100000 steps. Start with few runs, this one is slow.
        throw new NotImplementedException();
    }
}
