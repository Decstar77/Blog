
namespace arcade {
    public interface PPOInterface<TState, TAction> {
        public double[]             InitialActor();
        public double[]             InitialCritic();
        public void                 NewGame();
        public TState               GetState();
        public TAction              Think( TState s, double[] theta );   // sample a ~ pi(.|s, theta)
        public float                Step( TAction a );                   // apply a, return the reward
        public bool                 IsTerminal();
        public void                 AccumulateGradLogProb( TState s, TAction a, double[] theta, double scale, double[] into ); // into += scale * grad_theta ln pi(a|s, theta)

        // Logging
        public void EndEpisode( double episodeReward );
        public void PrintLog( int episode, int total, double seconds );
    }

    public struct PPOSettings {
        public double gamma;
        public double lambda;
        public double clipping;
        public double entropy;
        public double alphaActor;
        public double alphaCritic;
        public int rollOut;
        public int epochs;
        public int batchSize;
    }

    public class PPO {
        public PPO() {

        }

        public (double[], double[]) PolicyControl<TState, TAction>( PPOInterface<TState, TAction> ppo, PPOSettings settings ) {
            double[] actor = ppo.InitialActor();
            double[] critic = ppo.InitialCritic();


            return (actor, critic);
        }
    }
}
