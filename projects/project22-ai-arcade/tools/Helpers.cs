
namespace arcade {
    public static class Helpers {
        public static int Choose( int Count, Func<int, double> selector ) {
            double roll = Random.Shared.NextDouble();
            double cumulativeSum = 0.0;

            for ( int i = 0; i < Count; i++ ) {
                cumulativeSum += selector( i );
                if ( roll < cumulativeSum ) {
                    return i;
                }
            }

            return Count - 1;
        }

        public static int ArgMax( int Count, Func<int, double> selector ) {
            int maxAction = 0;
            double maxValue = double.MinValue;
            for ( int i = 0; i < Count; i++ ) {
                double v = selector( i );
                if ( v > maxValue ) {
                    maxValue = v;
                    maxAction = i;
                }
            }
            
            return maxAction;
        }
    }
}

