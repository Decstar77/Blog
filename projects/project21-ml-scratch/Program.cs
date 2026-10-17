
namespace nano {

    public class Value {
        private char op  = '\0';
        private double v = 0;
        private List<Value> children = new List<Value>();

        public Value( double v, char op = '\0') {
            this.v = v;
            this.op = op;
        }

        public Value( double v, List<Value> children, char op = '\0' ) : this( v, op ) {
            this.children = children;
        }

        public static Value operator +( Value x, Value y ) {
            return new Value( x.v + y.v, new List<Value>{ x, y }, '+' );
        }

        public static Value operator -( Value x, Value y ) {
            return new Value( x.v - y.v, new List<Value> { x, y }, '-' );
        }

        public static Value operator *( Value x, Value y ) {
            return new Value( x.v * y.v, new List<Value> { x, y }, '*' );
        }

        public static Value operator /( Value x, Value y ) {
            return new Value( x.v / y.v, new List<Value> { x, y }, '/' );
        }

        public override string ToString() {
            return v.ToString();
        }
    }

    class Program {
        public static void Main( string[] args ) {
            Value v1 = new Value( 4 );
            Value v2 = new Value( 3 );
            Value v3 = v1 + v2;
            Console.WriteLine( v3 );
            Console.ReadLine();
        }
    }
}
