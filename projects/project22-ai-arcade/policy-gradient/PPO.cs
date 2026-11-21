
namespace arcade {
    /*
    ===============================================================================
    ===============================================================================
    */
    public interface PPOInterface<TState, TAction> {
        // Parameters
        public double[]             InitialActor();
        public double[]             InitialCritic();

        // Environment
        public void                 NewGame();
        public TState               GetState();
        public float                Step( TAction a );                   // apply a, return the reward
        public bool                 IsTerminal();

        // Actor
        public (TAction, double)    Think( TState s, double[] theta );   // sample a ~ pi(.|s, theta), return (a, ln pi(a|s, theta))
        public double               LogProb( TState s, TAction a, double[] theta );                                     // ln pi(a|s, theta)
        public void                 AccumulateGradLogProb( TState s, TAction a, double[] theta, double scale, double[] into ); // into += scale * grad_theta ln pi(a|s, theta)
        public double               Entropy( TState s, double[] theta );                                                // H(pi(.|s, theta))
        public void                 AccumulateGradEntropy( TState s, double[] theta, double scale, double[] into );     // into += scale * grad_theta H(pi(.|s, theta))

        // Critic
        public double               Value( TState s, double[] w );                                                      // V(s, w)
        public void                 AccumulateGradValue( TState s, double[] w, double scale, double[] into );           // into += scale * grad_w V(s, w)

        // Logging
        public void EndEpisode( double episodeReward );
        public void EndUpdate( PPOStats stats );
        public void PrintLog( int iteration, int total, double seconds );
    }

    /*
    ===============================================================================
    ===============================================================================
    */
    public struct PPOSettings {
        public int    iterations;          // number of (rollout, update) cycles
        public int    rollOut;             // env steps collected per iteration (spans episode boundaries)
        public int    epochs;              // passes over each rollout
        public int    batchSize;           // minibatch size within an epoch
        public double gamma;               // discount
        public double lambda;              // GAE lambda
        public double clipping;            // epsilon in clip(ratio, 1 - eps, 1 + eps)
        public double entropy;             // entropy bonus coefficient
        public double alphaActor;
        public double alphaCritic;
        public double maxGradNorm;         // global grad-norm clip per network, <= 0 disables
        public double targetKL;            // stop the epoch loop early once approx KL exceeds 1.5x this, <= 0 disables
        public int?   seed;
        public int    logEvery;            // iterations between PrintLog calls, <= 0 disables

        public static PPOSettings Default => new PPOSettings {
            iterations          = 500,
            rollOut             = 2048,
            epochs              = 10,
            batchSize           = 64,
            gamma               = 0.99,
            lambda              = 0.95,
            clipping            = 0.2,
            entropy             = 0.01,
            alphaActor          = 3e-4,
            alphaCritic         = 1e-3,
            maxGradNorm         = 0.5,
            targetKL            = 0.0,
            seed                = null,
            logEvery            = 10,
        };
    }

    /*
    ===============================================================================
    ===============================================================================
    */
    public struct PPOStats {
        public int    iteration;
        public int    episodesFinished;    // episodes that ended during this iteration's rollout
        public int    epochsRun;           // < settings.epochs if the KL early-stop kicked in
        public double policyObjective;     // mean clipped surrogate, last epoch
        public double valueLoss;           // mean 0.5 * (V - R)^2, last epoch
        public double entropy;             // mean policy entropy, last epoch
        public double approxKL;            // mean KL(old || new) estimate, last epoch
        public double clipFraction;        // fraction of samples whose ratio was clipped, last epoch
    }

    /*
    ===============================================================================
    ===============================================================================
    */
    public class PPO {
        struct Sample<TState, TAction> {
            public TState  s;
            public TAction a;
            public double  logProb;   // ln pi_old(a|s)
            public double  value;     // V_old(s)
            public float   r;
            public bool    done;      // episode terminated after this step
            public double  advantage;
            public double  ret;       // advantage + value, the critic's regression target
        }

        /*
        ==========================
        ==========================
         */
        public PPO() {
        }

        /*
        ==========================
        ==========================
         */
        public (double[], double[]) PolicyControl<TState, TAction>( PPOInterface<TState, TAction> ppo, PPOSettings settings ) {
            var clock = System.Diagnostics.Stopwatch.StartNew();
            Random rng = settings.seed != null ? new Random( settings.seed.Value ) : new Random();

            double[] actor  = ppo.InitialActor();
            double[] critic = ppo.InitialCritic();
            double[] gActor  = new double[actor.Length];
            double[] gCritic = new double[critic.Length];

            int T = settings.rollOut;
            int batchSize = Math.Clamp( settings.batchSize, 1, T );
            var buffer = new Sample<TState, TAction>[T];
            int[] order = new int[T];
            for ( int i = 0; i < T; i++ ) order[i] = i;

            ppo.NewGame();
            double episodeReward = 0;

            for ( int iteration = 0; iteration < settings.iterations; iteration++ ) {
                PPOStats stats = new PPOStats { iteration = iteration + 1 };

                // Rollout ================
                // Games carry over between iterations; only terminal states trigger a reset.
                for ( int t = 0; t < T; t++ ) {
                    TState s = ppo.GetState();
                    (TAction a, double logProb) = ppo.Think( s, actor );
                    double v = ppo.Value( s, critic );
                    float  r = ppo.Step( a );
                    bool done = ppo.IsTerminal();

                    buffer[t] = new Sample<TState, TAction> { s = s, a = a, logProb = logProb, value = v, r = r, done = done };

                    episodeReward += r;
                    if ( done ) {
                        ppo.EndEpisode( episodeReward );
                        stats.episodesFinished++;
                        episodeReward = 0;
                        ppo.NewGame();
                    }
                }

                // Bootstrap from the state we stopped in. If the last step was terminal the
                // done mask zeroes this out, so it doesn't matter that it's a fresh game.
                double lastValue = ppo.Value( ppo.GetState(), critic );

                // GAE ================
                double gae = 0;
                for ( int t = T - 1; t >= 0; t-- ) {
                    double nextValue   = t == T - 1 ? lastValue : buffer[t + 1].value;
                    double nonTerminal = buffer[t].done ? 0.0 : 1.0;
                    double delta = buffer[t].r + settings.gamma * nextValue * nonTerminal - buffer[t].value;
                    gae = delta + settings.gamma * settings.lambda * nonTerminal * gae;
                    buffer[t].advantage = gae;
                    buffer[t].ret       = gae + buffer[t].value;
                }

                // Update ================
                for ( int epoch = 0; epoch < settings.epochs; epoch++ ) {
                    Shuffle( order, rng );

                    double policySum = 0;
                    double valueSum = 0;
                    double entropySum = 0;
                    double klSum = 0;
                    int clipped = 0;

                    for ( int start = 0; start < T; start += batchSize ) {
                        int end = Math.Min( start + batchSize, T );
                        double invN = 1.0 / ( end - start );
                        Array.Clear( gActor );
                        Array.Clear( gCritic );

                        for ( int k = start; k < end; k++ ) {
                            ref Sample<TState, TAction> x = ref buffer[order[k]];

                            // Actor: ascend L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) + c * H
                            double logRatio = ppo.LogProb( x.s, x.a, actor ) - x.logProb;
                            double ratio    = Math.Exp( logRatio );
                            double A        = x.advantage;
                            double ratioClipped = Math.Clamp( ratio, 1.0 - settings.clipping, 1.0 + settings.clipping );
                            policySum += Math.Min( ratio * A, ratioClipped * A );

                            // The min picks the clipped branch (zero gradient) exactly when the ratio
                            // has already moved past the clip boundary in the direction A favours.
                            bool isClipped = ( A >= 0 && ratio > 1.0 + settings.clipping ) || ( A < 0 && ratio < 1.0 - settings.clipping );
                            if ( isClipped ) {
                                clipped++;
                            }
                            else {
                                // grad(ratio) = ratio * grad ln pi
                                ppo.AccumulateGradLogProb( x.s, x.a, actor, ratio * A * invN, gActor );
                            }

                            entropySum += ppo.Entropy( x.s, actor );
                            if ( settings.entropy != 0 ) {
                                ppo.AccumulateGradEntropy( x.s, actor, settings.entropy * invN, gActor );
                            }

                            // http://joschu.net/blog/kl-approx.html (k3), always >= 0
                            klSum += ( ratio - 1.0 ) - logRatio;

                            // Critic: descend 0.5 * (V - R)^2  =>  w += alpha * (R - V) * grad V
                            double v = ppo.Value( x.s, critic );
                            double err = x.ret - v;
                            valueSum += 0.5 * err * err;
                            ppo.AccumulateGradValue( x.s, critic, err * invN, gCritic );
                        }

                        ClipGradNorm( gActor, settings.maxGradNorm );
                        ClipGradNorm( gCritic, settings.maxGradNorm );
                        for ( int i = 0; i < actor.Length; i++ )  actor[i]  += settings.alphaActor  * gActor[i];
                        for ( int i = 0; i < critic.Length; i++ ) critic[i] += settings.alphaCritic * gCritic[i];
                    }

                    stats.epochsRun       = epoch + 1;
                    stats.policyObjective = policySum / T;
                    stats.valueLoss       = valueSum / T;
                    stats.entropy         = entropySum / T;
                    stats.approxKL        = klSum / T;
                    stats.clipFraction    = (double)clipped / T;

                    if ( settings.targetKL > 0 && stats.approxKL > 1.5 * settings.targetKL ) {
                        break;
                    }
                }

                // Logging ================
                ppo.EndUpdate( stats );
                if ( settings.logEvery > 0 && ( iteration + 1 ) % settings.logEvery == 0 ) {
                    ppo.PrintLog( iteration + 1, settings.iterations, clock.Elapsed.TotalSeconds );
                }
            }

            return (actor, critic);
        }

        /*
        ==========================
        ==========================
         */
        private static void Shuffle( int[] a, Random rng ) {
            for ( int i = a.Length - 1; i > 0; i-- ) {
                int j = rng.Next( i + 1 );
                (a[i], a[j]) = (a[j], a[i]);
            }
        }

        /*
        ==========================
        ==========================
         */
        private static void ClipGradNorm( double[] g, double maxNorm ) {
            if ( maxNorm <= 0 ) return;
            double sq = 0;
            for ( int i = 0; i < g.Length; i++ ) sq += g[i] * g[i];
            double norm = Math.Sqrt( sq );
            if ( norm <= maxNorm ) return;
            double scale = maxNorm / norm;
            for ( int i = 0; i < g.Length; i++ ) g[i] *= scale;
        }
    }
}
