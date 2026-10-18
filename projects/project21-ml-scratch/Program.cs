
using System.Diagnostics;

namespace nano {

    public class Value {
        private char         op  = '\0';
        private double       v = 0;
        private double       grad = 0;
        private List<Value>  children = new List<Value>();
        private Action?      backward = null;

        public Value( double v, char op = '\0' ) {
            this.v = v;
            this.op = op;
        }

        public Value( double v, List<Value> children, char op = '\0' ) : this( v, op ) {
            this.children = children;
        }

        public double GetValue() { return v; }

        private void BuildTopo( Value value, List<Value> topo, HashSet<Value> visited ) {
            if ( visited.Contains( value ) == false ) {
                visited.Add( value );
                foreach ( Value child in value.children ) {
                    BuildTopo( child, topo, visited );
                }
                topo.Add( value );
            }
        }

        public void ZeroGrad() {
            grad = 0;
        }

        public void Update(double lr) {
            v -= lr * grad;
        }

        public void Backward() {
            List<Value> topo = new List<Value>();
            HashSet<Value> visited = new HashSet<Value>();
            BuildTopo( this, topo, visited );
            grad = 1;
            topo.Reverse();
            foreach ( Value child in topo ) {
                if ( child.backward != null ) {
                    child.backward();
                }
            }
        }

        public static Value operator +( Value x, Value y ) {
            Value v = new Value( x.v + y.v, new List<Value>{ x, y }, '+' );
            v.backward = () => {
                // f(a) = a + b -> f'(a) = 1 with chain rule 1 * XX
                x.grad += 1.0 * v.grad;
                y.grad += 1.0 * v.grad;
            };
            return v;
        }

        public static Value operator -( Value x, Value y ) {
            Value v = new Value( x.v - y.v, new List<Value>{ x, y }, '-' );
            v.backward = () => {
                // f(a) = a + b -> f'(a) = +-1 with chain rule 1 * XX
                x.grad += 1.0 * v.grad;
                y.grad -= 1.0 * v.grad;
            };
            return v;
        }

        public static Value operator *( Value x, Value y ) {
            Value v = new Value( x.v * y.v, new List<Value>{ x, y }, '*' );
            v.backward = () => {
                // f(a) = a * b -> f'(a) = b with chain rule b * XX
                x.grad += y.v * v.grad;
                y.grad += x.v * v.grad;
            };
            return v;
        }

        public static Value operator /( Value x, Value y ) {
            Value v = new Value( x.v / y.v, new List<Value>{ x, y }, '/' );
            v.backward = () => {
                // f(a) = a / b -> f'(a) = 1/b with chain rule b * XX
                x.grad += 1.0 / y.v * v.grad;
                y.grad += -x.v / ( y.v * y.v ) * v.grad;
            };
            return v;
        }

        public static Value Pow( Value x, double y ) {
            Value v = new Value( Math.Pow(x.v, y), new List<Value>{ x }, '^' );
            v.backward = () => {
                x.grad += ( y * Math.Pow( x.v, ( y - 1 ) ) ) * v.grad;
            };
            return v;
        }

        public static Value Relu( Value x ) {
            Value v = new Value( x.v < 0 ? 0 : x.v, new List<Value>{x}, 'R');
            v.backward = () => {
                x.grad += ( v.v > 0 ? 1.0 : 0.0 ) * v.grad;
            };
            return v;
        }

        public override string ToString() {
            return v.ToString();
        }
    }

    class Layer {
        public Value[] weights = new Value[0];
        public Value[] biases = new Value[0];
        public int inCount = 0;
        public int outCount = 0;
        public int weightCount = 0;
        public bool act = true;
        public Layer( int inCount, int outCount, bool act = true ) {
            this.inCount = inCount;
            this.outCount = outCount;
            this.weightCount = inCount * outCount;
            this.act = act;

            weights = new Value[weightCount];
            for ( int i = 0; i < weightCount; i++ ) { weights[i] = new Value( Random.Shared.NextDouble() * 2 - 1 ); }
            biases = new Value[outCount];
            for ( int i  = 0; i < outCount;i++ ) { biases[i] = new Value(Random.Shared.NextDouble() * 2 - 1); }
        }

        public Value[] Forward( Value[] inputs ) {
            Debug.Assert( inCount == inputs.Length );

            Value[] outputs = new Value[outCount];
            for ( int neuronIndex = 0; neuronIndex < outCount; neuronIndex++ ) {
                outputs[neuronIndex] = new Value( 0 );
                for ( int inputIndex = 0; inputIndex < inCount; inputIndex++ ) {
                    int weightIndex = neuronIndex * inCount + inputIndex;
                    outputs[neuronIndex] += weights[weightIndex] * inputs[inputIndex];
                }
                outputs[neuronIndex] += biases[neuronIndex];
            }

            if ( act == true ) {
                for ( int i = 0; i < outCount; i++ ) {
                    outputs[i] = Value.Relu( outputs[i] );
                }
            }

            return outputs;
        }

        public void ZeroGrad() {
            for ( int i = 0; i < weightCount; i++ ) { weights[i].ZeroGrad(); }
            for ( int i = 0; i < outCount; i++ ) { biases[i].ZeroGrad();  }
        }

        public void Update(double lr) {
            for ( int i = 0; i < weightCount; i++ ) { weights[i].Update(lr); }
            for ( int i = 0; i < outCount; i++ ) { biases[i].Update(lr); }
        }
    }

    class Program {
        public static Random random = new Random();

        public static void Test1() {
            Value i1 = new Value(2);
            Value i2 = new Value(2);
            Value w1 = new Value(1);
            Value w2 = new Value(1);
            Value b = new Value(1);
            Value a = new Value(8);

            Value v1 = i1 * w1 + i2 * w2 + b;
            Value l = Value.Pow(a - v1, 2);
            Console.WriteLine( l );

            l.Backward();
            double lr = 0.1;
            w1.Update( lr );
            w2.Update( lr );
            b.Update( lr );

            v1 = i1 * w1 + i2 * w2 + b;
            l = Value.Pow( a - v1, 2 );
            Console.WriteLine( l );
        }

        public static void Main( string[] args ) {
            Layer layer1 = new Layer(1, 32);
            Layer layer2 = new Layer(32, 32);
            Layer layer3 = new Layer(32, 1, false);

            const int steps = 100_000;
            double avgLoss = 0;
            double lr = 0.01;

            for ( int i = 0; i < steps; i++ ) {
                double x = random.NextDouble();
                double y = x * x;
                Value[] input = { new Value(x) };
                Value[] answers = { new Value (y) };

                Value[] l1 = layer1.Forward( input );
                Value[] l2 = layer2.Forward( l1 );
                Value[] l3 = layer3.Forward( l2 );

                Value loss = Value.Pow( answers[0] - l3[0], 2 );
                double lv = loss.GetValue();

                layer1.ZeroGrad();
                layer2.ZeroGrad();
                layer3.ZeroGrad();
                loss.Backward();
                layer1.Update( lr );
                layer2.Update( lr );
                layer3.Update( lr );

                avgLoss = (i * avgLoss + lv) / ( i + 1 );

                Console.Write( $"\rLoss={avgLoss:F2}");
            }
            
            Console.ReadLine();
        }
    }
}

