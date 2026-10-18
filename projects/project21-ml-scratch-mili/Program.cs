
namespace mili {
    static class ActivationFunctions {
        public static double Relu( double x ) {
            return x > 0 ? x : 0;
        }
        public static double ReluPrime( double x ) {
            return x > 0 ? 1 : 0;
        }
    }

    class Layer {
        private int inCount = 0;
        private int outCount = 0;
        private int weightCount = 0;
        private bool act = true;
        private double[] weights = Array.Empty<double>();
        private double[] weightGrads = Array.Empty<double>();
        private double[] biases = Array.Empty<double>();
        private double[] biasGrads = Array.Empty<double>();
        private double[] lastInputs = Array.Empty<double>();
        private double[] lastZ = Array.Empty<double>();

        public Layer( int inCount, int outCount, bool act = true ) {
            this.inCount = inCount;
            this.outCount = outCount;
            this.weightCount = inCount * outCount;
            this.act = act;

            weights = new double[weightCount];
            weightGrads = new double[weightCount];
            biases = new double[outCount];
            biasGrads = new double[outCount];

            for ( int i = 0; i < weightCount; i++ ) { weights[i] = Random.Shared.NextDouble() * 2 - 1; }
            for ( int i = 0; i < outCount; i++ ) { biases[i] = Random.Shared.NextDouble() * 2 - 1; }
        }

        public double[] Forward( double[] inputs ) {
            if ( inputs.Length != inCount ) { throw new ArgumentOutOfRangeException(); }

            lastInputs = inputs;
            lastZ = new double[outCount];
            double[] outputs = new double[outCount];
            for ( int neuronIndex = 0; neuronIndex < outCount; neuronIndex++ ) {
                double z = biases[neuronIndex];
                for ( int edgeIndex = 0; edgeIndex < inCount; edgeIndex++ ) {
                    int weightIndex = neuronIndex * inCount + edgeIndex;
                    z += weights[weightIndex] * inputs[edgeIndex];
                }

                lastZ[neuronIndex] = z;
                outputs[neuronIndex] = act ? ActivationFunctions.Relu( z ) : z;
            }

            return outputs;
        }

        public void ZeroGrad() {
            for ( int i = 0; i < weightCount; i++ ) { weightGrads[i] = 0; }
            for ( int i = 0; i < outCount; i++ ) { biasGrads[i] = 0; }
        }

        public void Update( double lr ) {
            for ( int neuronIndex = 0; neuronIndex < outCount; neuronIndex++ ) {
                biases[neuronIndex] -= lr * biasGrads[neuronIndex];
                for ( int edgeIndex = 0; edgeIndex < inCount; edgeIndex++ ) {
                    int weightIndex = neuronIndex * inCount + edgeIndex;
                    weights[weightIndex] -= lr * weightGrads[weightIndex];
                }
            }
        }

        public double[] Backward( double[] fGrads ) {
            if ( fGrads.Length != outCount ) { throw new ArgumentOutOfRangeException(); }

            double[] outGrads = new double[inCount];
            for ( int neuronIndex = 0; neuronIndex < outCount; neuronIndex++ ) {
                double dz = act ? ActivationFunctions.ReluPrime( lastZ[neuronIndex] ) * fGrads[neuronIndex] : fGrads[neuronIndex];

                biasGrads[neuronIndex] += dz;
                for ( int edgeIndex = 0; edgeIndex < inCount; edgeIndex++ ) {
                    int weightIndex = neuronIndex * inCount + edgeIndex;
                    weightGrads[weightIndex] += dz * lastInputs[edgeIndex];
                    outGrads[edgeIndex] += weights[weightIndex] * dz;
                }
            }

            return outGrads;
        }
    }

    class Program {
        public static void Main( string[] args ) {
            Layer layer1 = new Layer(1, 32);
            Layer layer2 = new Layer(32, 32);
            Layer layer3 = new Layer(32, 1, false);

            const double lr = 0.01;
            const int trainingSteps = 1_000_000;
            double avgLoss = 0;
            for ( int i = 0; i < trainingSteps; i++ ) {
                double x = Random.Shared.NextDouble() * 2 - 1;
                double y = x * x;
                double[] l1 = layer1.Forward( new double []{ x } );
                double[] l2 = layer2.Forward(l1);
                double[] l3 = layer3.Forward(l2);
                double loss = Math.Pow(y - l3[0], 2);
                double dl = -2 * (y - l3[0]);

                layer3.ZeroGrad();
                layer2.ZeroGrad();
                layer1.ZeroGrad();

                double[] d3 = layer3.Backward( new double[] { dl } );
                double[] d2 = layer2.Backward(d3);
                double[] d1 = layer1.Backward(d2);

                layer3.Update( lr );
                layer2.Update( lr );
                layer1.Update( lr );

                avgLoss = (i * avgLoss + loss) / ( i + 1 );

                Console.Write( $"\rLoss={avgLoss:F5}" );
            }

            Console.ReadLine();
        }
    };
}
