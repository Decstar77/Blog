using System;
using System.Collections.Generic;
using System.Text;

namespace arcade {
    public static class ActivationFuncs {
        public static double Relu( double x) {
            return x > 0 ? x : 0;
        }
    }
}
