namespace arcade {
    /*
        Everything the actor-critic methods need from a game + function approximator.

        Same idea as PPOInterface, minus the PPO-only parts (LogProb, Entropy). Both algorithms below
        are fully online: they update after every single step, so nothing is stored and GetState() is
        free to return a reference into live game memory as long as it's read before Step() is called.
    */
    public interface ActorCriticInterface<TState, TAction> {
        // Parameters
        public double[]     InitialActor();
        public double[]     InitialCritic();

        // Environment
        public void         NewGame();
        public TState       GetState();
        public float        Step( TAction a );                   // apply a, return the reward
        public bool         IsTerminal();

        // Actor
        public TAction      Think( TState s, double[] theta );   // sample a ~ pi(.|s, theta)
        public void         AccumulateGradLogProb( TState s, TAction a, double[] theta, double scale, double[] into ); // into += scale * grad_theta ln pi(a|s, theta)

        // Critic
        public double       Value( TState s, double[] w );                                              // V(s, w)
        public void         AccumulateGradValue( TState s, double[] w, double scale, double[] into );   // into += scale * grad_w V(s, w)

        // Logging
        public void         EndEpisode( double episodeReward );
        public void         PrintLog( int episode, int total, double seconds );
    }

    public struct ActorCriticSettings {
        public int    episodes;
        public double gamma;
        public double alphaActor;
        public double alphaCritic;
        public double lambdaActor;           // trace decay for the actor  (ActorCriticEligibilityTraces only)
        public double lambdaCritic;          // trace decay for the critic (ActorCriticEligibilityTraces only)

        // Sutton & Barto multiply the actor update by I = gamma^t, which makes it a correct gradient of the
        // discounted objective from the start state. In long episodes (lots of game ticks) I shrinks to ~0
        // and the actor stops learning anything late in the episode, so most practical implementations
        // drop it. Off by default.
        public bool   discountActorUpdates;

        public int    logEvery;              // episodes between PrintLog calls, <= 0 disables

        public static ActorCriticSettings Default => new ActorCriticSettings {
            episodes             = 1000,
            gamma                = 0.99,
            alphaActor           = 1e-4,
            alphaCritic          = 1e-3,
            lambdaActor          = 0.9,
            lambdaCritic         = 0.9,
            discountActorUpdates = false,
            logEvery             = 10,
        };
    }

    /*
        One-step actor-critic (Sutton & Barto, 2nd ed., section 13.5).

        Like REINFORCE with a baseline, but instead of waiting for the full return G_t at the end of the
        episode, the critic's estimate stands in for everything after the next reward:

            delta = R + gamma * V(S') - V(S)        (TD error, the one-step advantage estimate)
            w     += alphaCritic * delta * grad V(S)
            theta += alphaActor  * delta * grad ln pi(A|S)
    */
    public class ActorCritic {
        public (double[], double[]) PolicyControl<TState, TAction>( ActorCriticInterface<TState, TAction> ac, ActorCriticSettings settings ) {
            var clock = System.Diagnostics.Stopwatch.StartNew();

            double[] actor  = ac.InitialActor();
            double[] critic = ac.InitialCritic();
            double[] gradLogProb = new double[actor.Length];
            double[] gradValue   = new double[critic.Length];

            for ( int episodeIndex = 0; episodeIndex < settings.episodes; episodeIndex++ ) {
                ac.NewGame();
                double I = 1.0;
                double episodeReward = 0;

                while ( ac.IsTerminal() == false ) {
                    // Act ================
                    TState  s = ac.GetState();
                    TAction a = ac.Think( s, actor );
                    double  v = ac.Value( s, critic );   // must be read before Step() changes the game
                    float   r = ac.Step( a );
                    episodeReward += r;

                    // TD error ================
                    // A terminal state has no future, so its value is 0 by definition (not whatever the critic says).
                    double nextValue = ac.IsTerminal() ? 0.0 : ac.Value( ac.GetState(), critic );
                    double delta = r + settings.gamma * nextValue - v;

                    // Gradients ================
                    // Both are taken at the pre-update weights, before either network changes.
                    Array.Clear( gradLogProb );
                    Array.Clear( gradValue );
                    ac.AccumulateGradLogProb( s, a, actor, 1.0, gradLogProb );
                    ac.AccumulateGradValue( s, critic, 1.0, gradValue );

                    // Update ================
                    double actorScale = settings.discountActorUpdates ? I : 1.0;
                    for ( int i = 0; i < critic.Length; i++ ) critic[i] += settings.alphaCritic * delta * gradValue[i];
                    for ( int i = 0; i < actor.Length; i++ )  actor[i]  += settings.alphaActor * actorScale * delta * gradLogProb[i];

                    I *= settings.gamma;
                }

                // Logging ================
                ac.EndEpisode( episodeReward );
                if ( settings.logEvery > 0 && ( episodeIndex + 1 ) % settings.logEvery == 0 ) {
                    ac.PrintLog( episodeIndex + 1, settings.episodes, clock.Elapsed.TotalSeconds );
                }
            }

            return (actor, critic);
        }
    }

    /*
        Actor-critic with eligibility traces (Sutton & Barto, 2nd ed., section 13.5, episodic).

        Same TD error as one-step actor-critic, but instead of applying delta only to the gradient of the
        current step, each network keeps a trace z: a decaying sum of recent gradients. A surprise (large
        delta) now also adjusts the states/actions that led up to it, not just the last one.

            z_w     = gamma * lambdaCritic * z_w     + grad V(S)
            z_theta = gamma * lambdaActor  * z_theta + grad ln pi(A|S)
            w      += alphaCritic * delta * z_w
            theta  += alphaActor  * delta * z_theta

        lambda = 0 gives back one-step actor-critic. lambda -> 1 approaches REINFORCE with a baseline.
    */
    public class ActorCriticEligibilityTraces {
        public (double[], double[]) PolicyControl<TState, TAction>( ActorCriticInterface<TState, TAction> ac, ActorCriticSettings settings ) {
            var clock = System.Diagnostics.Stopwatch.StartNew();

            double[] actor  = ac.InitialActor();
            double[] critic = ac.InitialCritic();
            double[] traceActor  = new double[actor.Length];
            double[] traceCritic = new double[critic.Length];

            for ( int episodeIndex = 0; episodeIndex < settings.episodes; episodeIndex++ ) {
                ac.NewGame();
                Array.Clear( traceActor );    // traces never carry across episodes
                Array.Clear( traceCritic );
                double I = 1.0;
                double episodeReward = 0;

                while ( ac.IsTerminal() == false ) {
                    // Act ================
                    TState  s = ac.GetState();
                    TAction a = ac.Think( s, actor );
                    double  v = ac.Value( s, critic );
                    float   r = ac.Step( a );
                    episodeReward += r;

                    // TD error ================
                    double nextValue = ac.IsTerminal() ? 0.0 : ac.Value( ac.GetState(), critic );
                    double delta = r + settings.gamma * nextValue - v;

                    // Traces ================
                    // Decay the old trace, then add this step's gradient on top.
                    double actorScale = settings.discountActorUpdates ? I : 1.0;
                    for ( int i = 0; i < critic.Length; i++ ) traceCritic[i] *= settings.gamma * settings.lambdaCritic;
                    for ( int i = 0; i < actor.Length; i++ )  traceActor[i]  *= settings.gamma * settings.lambdaActor;
                    ac.AccumulateGradValue( s, critic, 1.0, traceCritic );
                    ac.AccumulateGradLogProb( s, a, actor, actorScale, traceActor );

                    // Update ================
                    for ( int i = 0; i < critic.Length; i++ ) critic[i] += settings.alphaCritic * delta * traceCritic[i];
                    for ( int i = 0; i < actor.Length; i++ )  actor[i]  += settings.alphaActor * delta * traceActor[i];

                    I *= settings.gamma;
                }

                // Logging ================
                ac.EndEpisode( episodeReward );
                if ( settings.logEvery > 0 && ( episodeIndex + 1 ) % settings.logEvery == 0 ) {
                    ac.PrintLog( episodeIndex + 1, settings.episodes, clock.Elapsed.TotalSeconds );
                }
            }

            return (actor, critic);
        }
    }
}
