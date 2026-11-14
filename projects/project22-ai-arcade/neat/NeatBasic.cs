namespace neat {
    /*
    ===============================================================================
    ===============================================================================
    */
    public sealed class NeatSettings {
        public int    PopulationSize              { get; init; } = 150;
        public double CompatibilityThreshold      { get; init; } = 3;
        public double ExcessCoefficient           { get; init; } = 1;
        public double DisjointCoefficient         { get; init; } = 1;
        public double WeightDifferenceCoefficient { get; init; } = 0.4;
        public double SurvivalFraction            { get; init; } = 0.2;
        public double CrossoverChance             { get; init; } = 0.75;
        public double WeightSelectionChance       { get; init; } = 0.8;
        public double WeightMutationChance        { get; init; } = 0.9;
        public double WeightPerturbScale          { get; init; } = 1;
        public double BiasSelectionChance         { get; init; } = 0.8;
        public double BiasMutationChance          { get; init; } = 0.9;
        public double BiasPerturbScale            { get; init; } = 1;
        public double ConnectionSelectionChance   { get; init; } = 0.08;
        public double NodeSelectionChance         { get; init; } = 0.03;
        public int    MaxStagnation               { get; init; } = 20;
    }

    /*
    ===============================================================================
    ===============================================================================
    */
    public static class RNG {
        public static Random random = new Random();

        public static void      Seed( int s )         { random = new Random( s ); }
        public static double    NextWeight()          { return random.NextDouble() * 2 - 1; }
        public static T         Choose<T>( T a, T b ) { return random.NextDouble() > 0.5 ? a : b; }
    }

    /*
    ===============================================================================
    ===============================================================================
    */
    public static class ActivationFunctions {
        public static double Sigmoid( double x ) { return 1.0 / ( 1.0 + Math.Exp( -x ) ); }
        public static double ReLu( double x )    { return x > 0 ? x : 0; }
    }

    /*
    ===============================================================================
    ===============================================================================
    */
    public enum GeneNodeType {
        Input,
        Hidden,
        Output
    }

    /*
    ===============================================================================
    ===============================================================================
    */
    public static class InnovationRegistry {
        public static int GlobalNodeIdCounter = 1;
        public static int GlobalInnovationCounter = 1;

        public static Dictionary<(int SrcId, int DstId), int>   connectionInnovations = new Dictionary<(int SrcId, int DstId), int>();
        public static Dictionary<int, int>                      splits = new Dictionary<int, int>();

        /*
        ==========================
        ==========================
         */
        public static GeneNode Split( int innovationNumber ) {
            if ( !splits.TryGetValue( innovationNumber, out int nodeId ) ) {
                nodeId = GlobalNodeIdCounter++;
                splits.Add( innovationNumber, nodeId );
            }
            return new GeneNode( GeneNodeType.Hidden, nodeId );
        }

        /*
        ==========================
        ==========================
         */
        public static GeneConnection MakeConnection( GeneNode src, GeneNode dst ) {
            if ( !connectionInnovations.TryGetValue( (src.nodeId, dst.nodeId), out int innovationNumber ) ) {
                innovationNumber = GlobalInnovationCounter++;
                connectionInnovations.Add( (src.nodeId, dst.nodeId), innovationNumber );
            }
            return new GeneConnection( src, dst, innovationNumber );
        }
    }

    /*
    ===============================================================================
    ===============================================================================
    */
    public class GeneNode {
        public int nodeId;
        public GeneNodeType type;
        public double value;
        public double bias;
        public int layer;
        /*
        ==========================
        ==========================
         */
        public GeneNode( GeneNodeType type, int id ) {
            nodeId = id;
            this.type = type;
        }
    }

    public class GeneConnection {
        public GeneNode src;
        public GeneNode dst;
        public double   weight;
        public bool     enabled;
        public int      innovationNumber;

        /*
        ==========================
        ==========================
         */
        public GeneConnection( GeneNode src, GeneNode dst, int innovationNumber ) {
            this.src = src;
            this.dst = dst;
            this.innovationNumber = innovationNumber;
            enabled = true;
            weight = RNG.NextWeight();
        }
    }

    /*
    ===============================================================================
    ===============================================================================
    */
    public class Genome {
        public List<GeneNode>       nodes = new List<GeneNode>();
        public List<GeneConnection> connections = new List<GeneConnection>();
        public double               fitness = 0;

        /*
        ==========================
        ==========================
         */
        public Genome Clone() {
            Genome clone = new Genome() { fitness = fitness };
            Dictionary<int, GeneNode> map = new Dictionary<int, GeneNode>();
            foreach ( GeneNode node in nodes ) {
                GeneNode copy = new GeneNode(node.type, node.nodeId)
                {
                    layer = node.layer,
                    bias = node.bias,
                    value = node.value
                };
                clone.nodes.Add( copy );
                map.Add( copy.nodeId, copy );
            }
            foreach ( GeneConnection connection in connections ) {
                clone.connections.Add( new GeneConnection( map[connection.src.nodeId], map[connection.dst.nodeId], connection.innovationNumber ) { weight = connection.weight, enabled = connection.enabled } );
            }
            return clone;
        }

        /*
        ==========================
        ==========================
         */
        public void Mutate( NeatSettings settings ) {
            WeightMutation( settings.WeightSelectionChance, settings.WeightMutationChance, settings.WeightPerturbScale );
            BiasMutation( settings.BiasSelectionChance, settings.BiasMutationChance, settings.BiasPerturbScale );
            ConnectionMutation( settings.ConnectionSelectionChance );
            NodeMutation( settings.NodeSelectionChance );
        }

        /*
        ==========================
        ==========================
         */
        public void WeightMutation( double selectionChance, double mutationChance, double perturbScale ) {
            if ( RNG.random.NextDouble() < selectionChance ) {
                for ( int c = 0; c < connections.Count; c++ ) {
                    if ( RNG.random.NextDouble() < mutationChance ) {
                        connections[c].weight += RNG.NextWeight() * perturbScale;
                    }
                    else {
                        connections[c].weight = RNG.NextWeight();
                    }
                }
            }
        }

        /*
        ==========================
        ==========================
         */
        public void BiasMutation( double selectionChance, double mutationChance, double perturbScale ) {
            if ( RNG.random.NextDouble() < selectionChance ) {
                foreach ( GeneNode node in nodes ) {
                    if ( node.type == GeneNodeType.Input ) continue;
                    if ( RNG.random.NextDouble() < mutationChance )
                        node.bias += RNG.NextWeight() * perturbScale;
                    else
                        node.bias = RNG.NextWeight();
                }
            }
        }

        /*
        ==========================
        ==========================
         */
        public void ConnectionMutation( double selectionChance ) {
            if ( RNG.random.NextDouble() < selectionChance ) {
                for ( int tries = 0; tries < 1000; tries++ ) {
                    int srcIndex = RNG.random.Next(nodes.Count);
                    int dstIndex = RNG.random.Next(nodes.Count);
                    GeneNode src = nodes[srcIndex];
                    GeneNode dst = nodes[dstIndex];

                    if ( srcIndex == dstIndex ) continue;
                    if ( dst.layer <= src.layer ) continue;
                    if ( dst.type == GeneNodeType.Input ) continue;
                    GeneConnection? existing = connections.Find(s => s.dst == dst && s.src == src);
                    if ( existing != null && existing.enabled == true ) continue;
                    if ( existing != null && existing.enabled == false ) {
                        existing.enabled = true;
                        break;
                    }
                    connections.Add( InnovationRegistry.MakeConnection( src, dst ) );
                    break;
                }
            }
        }

        /*
        ==========================
        ==========================
         */
        public void NodeMutation( double selectionChance ) {
            if ( RNG.random.NextDouble() < selectionChance ) {
                for ( int tries = 0; tries < 1000; tries++ ) {
                    int index = RNG.random.Next(connections.Count);
                    GeneConnection c = connections[index];
                    if ( c.enabled == false ) continue;

                    GeneNode node = InnovationRegistry.Split(c.innovationNumber);
                    if ( nodes.Find( x => x.nodeId == node.nodeId ) != null ) continue; // We've already split this connection

                    GeneConnection input = InnovationRegistry.MakeConnection(c.src, node);
                    GeneConnection output = InnovationRegistry.MakeConnection(node, c.dst);

                    c.enabled = false;
                    input.weight = 1;
                    output.weight = c.weight;

                    node.layer = c.src.layer + 1;
                    if ( c.dst.layer == node.layer ) {
                        for ( int n = 0; n < nodes.Count; n++ ) {
                            if ( nodes[n].layer >= node.layer ) {
                                nodes[n].layer++;
                            }
                        }
                    }

                    nodes.Add( node );
                    connections.Add( input );
                    connections.Add( output );

                    break;
                }
            }
        }

        /*
        ==========================
        ==========================
         */
        public double[] Forward( double[] inputs ) {
            int maxLayer = 0;
            int outputCount = 0;
            for ( int n = 0, i = 0; n < nodes.Count; n++ ) {
                if ( nodes[n].type == GeneNodeType.Input ) {
                    nodes[n].value = inputs[i++];
                }
                else if ( nodes[n].type == GeneNodeType.Output ) {
                    outputCount++;
                }
                maxLayer = Math.Max( nodes[n].layer, maxLayer );
            }

            double[] outputs = new double[outputCount];
            outputCount = 0;
            for ( int l = 0; l <= maxLayer; l++ ) {
                for ( int n = 0; n < nodes.Count; n++ ) {
                    if ( nodes[n].layer == l ) {
                        GeneNode node = nodes[n];
                        if ( node.type == GeneNodeType.Input ) continue;

                        double sum = node.bias;
                        for ( int c = 0; c < connections.Count; c++ ) {
                            if ( connections[c].enabled == true && connections[c].dst == node ) {
                                sum += connections[c].weight * connections[c].src.value;
                            }
                        }

                        node.value = sum;
                        if ( node.type == GeneNodeType.Output ) {
                            node.value = ActivationFunctions.Sigmoid( node.value );
                            outputs[outputCount++] = node.value;
                        }
                        else if ( node.type == GeneNodeType.Hidden ) {
                            node.value = ActivationFunctions.Sigmoid( node.value );
                        }
                    }
                }
            }
            return outputs;
        }

        /*
        ==========================
        ==========================
         */
        public static Genome Cross( Genome parentA, Genome parentB ) { // Assumes parentA is more fit than parentB
            Genome child = new Genome();
            Dictionary<int, GeneNode> map = new Dictionary<int, GeneNode>();
            foreach ( GeneNode n in parentA.nodes ) {
                GeneNode copy = new GeneNode(n.type, n.nodeId) { layer = n.layer, bias = n.bias };
                child.nodes.Add( copy );
                map[n.nodeId] = copy;
            }

            Dictionary<int, GeneConnection> bByInnov = parentB.connections.ToDictionary(c => c.innovationNumber);
            foreach ( GeneConnection a in parentA.connections ) {
                bByInnov.TryGetValue( a.innovationNumber, out GeneConnection? b );
                GeneConnection chosen = b != null ? RNG.Choose(a, b) : a;
                GeneConnection c = new GeneConnection(map[a.src.nodeId], map[a.dst.nodeId], a.innovationNumber) { weight = chosen.weight };
                bool disabledInEither = !a.enabled || (b != null && !b.enabled);
                c.enabled = !( disabledInEither && RNG.random.NextDouble() < 0.75 );
                child.connections.Add( c );
            }
            return child;
        }

        /*
        ==========================
        ==========================
         */
        public static int DisjointGenes( Genome a, Genome b ) {
            HashSet<int> aInnovations = a.connections.Select(c => c.innovationNumber).ToHashSet();
            HashSet<int> bInnovations = b.connections.Select(c => c.innovationNumber).ToHashSet();
            if ( aInnovations.Count == 0 || bInnovations.Count == 0 ) return 0;

            int sharedLimit = Math.Min(aInnovations.Max(), bInnovations.Max());
            return aInnovations.Count( i => i <= sharedLimit && !bInnovations.Contains( i ) ) + bInnovations.Count( i => i <= sharedLimit && !aInnovations.Contains( i ) );
        }

        /*
        ==========================
        ==========================
         */
        public static int ExcessGenes( Genome a, Genome b ) {
            if ( a.connections.Count == 0 ) return b.connections.Count;
            if ( b.connections.Count == 0 ) return a.connections.Count;

            int aMax = a.connections.Max(c => c.innovationNumber);
            int bMax = b.connections.Max(c => c.innovationNumber);
            return a.connections.Count( c => c.innovationNumber > bMax ) + b.connections.Count( c => c.innovationNumber > aMax );
        }

        /*
        ==========================
        ==========================
         */
        public static double AverageWeightDifference( Genome a, Genome b ) {
            Dictionary<int, GeneConnection> bByInnov = b.connections.ToDictionary(c => c.innovationNumber);
            double difference = 0;
            int matchingCount = 0;
            foreach ( GeneConnection connection in a.connections ) {
                if ( bByInnov.TryGetValue( connection.innovationNumber, out GeneConnection? matching ) ) {
                    difference += Math.Abs( connection.weight - matching.weight );
                    matchingCount++;
                }
            }
            return matchingCount == 0 ? 0 : difference / matchingCount;
        }

        /*
        ==========================
        ==========================
         */
        public static double Delta( Genome a, Genome b, double e, double d, double w ) {
            double N = Math.Max(1, Math.Max(a.connections.Count, b.connections.Count));
            double E = Genome.ExcessGenes(a, b);
            double D = Genome.DisjointGenes(a, b);
            double W = Genome.AverageWeightDifference(a, b);
            return ( e * E / N ) + ( d * D / N ) + ( w * W );
        }

        /*
        ==========================
        ==========================
         */
        public List<GeneNode>? TopologicalOrder() // null => cycle
        {
            var inDeg = nodes.ToDictionary(n => n.nodeId, _ => 0);
            foreach ( var c in connections ) inDeg[c.dst.nodeId]++;
            var q = new Queue<GeneNode>(nodes.Where(n => inDeg[n.nodeId] == 0));
            var order = new List<GeneNode>();
            while ( q.Count > 0 ) {
                var n = q.Dequeue();
                order.Add( n );
                foreach ( var c in connections )
                    if ( c.src.nodeId == n.nodeId && --inDeg[c.dst.nodeId] == 0 ) q.Enqueue( c.dst );
            }
            return order.Count == nodes.Count ? order : null;
        }

        /*
        ==========================
        ==========================
         */
        public double TestForward( double i1, double i2 ) {
            nodes[0].value = i1;
            nodes[1].value = i2;
            nodes[2].value = nodes[0].value * connections[0].weight + nodes[1].value * connections[1].weight + nodes[2].bias;
            nodes[2].value = ActivationFunctions.Sigmoid( nodes[2].value );
            return nodes[2].value;
        }
    }

    /*
    ===============================================================================
    ===============================================================================
    */
    public class Species {
        public List<Genome> members = new List<Genome>();
        public Genome       representative;
        public double       bestFitness = double.NegativeInfinity;
        public int          stagnation;

        /*
        ==========================
        ==========================
         */
        public Species( Genome representative ) { this.representative = representative.Clone(); }
    }

    /*
    ===============================================================================
    ===============================================================================
    */
    public sealed class NeatPopulation {
        private readonly NeatSettings   settings;
        private List<Genome>            genomes = new List<Genome>();
        private readonly List<Species>  species = new List<Species>();

        public IReadOnlyList<Genome>    Genomes => genomes.AsReadOnly();
        public int                      Generation { get; private set; }
        public int                      SpeciesCount => species.Count;
        public Genome?                  BestGenome { get; private set; }

        /*
        ==========================
        ==========================
         */
        public NeatPopulation( int inputCount, int outputCount, NeatSettings? settings = null ) {
            this.settings = settings ?? new NeatSettings();

            int[] inputIds = Enumerable.Range(0, inputCount).Select(_ => InnovationRegistry.GlobalNodeIdCounter++).ToArray();
            int[] outputIds = Enumerable.Range(0, outputCount).Select(_ => InnovationRegistry.GlobalNodeIdCounter++).ToArray();
            for ( int i = 0; i < this.settings.PopulationSize; i++ ) {
                Genome genome = new Genome();
                foreach ( int id in inputIds ) genome.nodes.Add( new GeneNode( GeneNodeType.Input, id ) );
                foreach ( int id in outputIds ) genome.nodes.Add( new GeneNode( GeneNodeType.Output, id ) { layer = 1, bias = RNG.NextWeight() } );
                foreach ( GeneNode input in genome.nodes.Where( n => n.type == GeneNodeType.Input ) )
                    foreach ( GeneNode output in genome.nodes.Where( n => n.type == GeneNodeType.Output ) )
                        genome.connections.Add( InnovationRegistry.MakeConnection( input, output ) );
                genomes.Add( genome );
            }
            Speciate();
        }

        /*
        ==========================
        ==========================
         */
        public void Evaluate( Func<Genome, double> fitnessFunction ) {
            foreach ( Genome genome in genomes ) {
                double fitness = fitnessFunction(genome);
                genome.fitness = fitness;
            }

            Genome champion = genomes.MaxBy(g => g.fitness)!;
            if ( BestGenome == null || champion.fitness > BestGenome.fitness ) BestGenome = champion.Clone();
            foreach ( Species group in species ) {
                double fitness = group.members.Max(g => g.fitness);
                if ( fitness > group.bestFitness ) {
                    group.bestFitness = fitness;
                    group.stagnation = 0;
                }
                else {
                    group.stagnation++;
                }
            }
        }

        /*
        ==========================
        ==========================
         */
        private void Speciate() {
            foreach ( Species group in species ) group.members.Clear();
            foreach ( Genome genome in genomes ) {
                Species? group = species.FirstOrDefault(s => Genome.Delta(genome, s.representative, settings.ExcessCoefficient, settings.DisjointCoefficient, settings.WeightDifferenceCoefficient) < settings.CompatibilityThreshold);
                if ( group == null ) {
                    group = new Species( genome );
                    species.Add( group );
                }
                group.members.Add( genome );
            }
            species.RemoveAll( s => s.members.Count == 0 );
            foreach ( Species group in species ) {
                group.representative = group.members[RNG.random.Next( group.members.Count )].Clone();
            }
        }

        /*
        ==========================
        ==========================
         */
        public void Evolve() {
            Genome champion = genomes.MaxBy(g => g.fitness)!;
            species.RemoveAll( s => s.stagnation >= settings.MaxStagnation && !s.members.Contains( champion ) );

            // Dividing each member's fitness by species size is explicit fitness sharing.
            double maxFitness = champion.fitness;
            var scores = species.Select(s => maxFitness == 0 ? 1 : s.members.Average(g => g.fitness / maxFitness)).ToArray();
            double scoreSum = scores.Sum();
            int remaining = settings.PopulationSize - species.Count;
            var allocations = scores.Select(score => remaining * score / scoreSum).ToArray();
            var counts = allocations.Select(a => 1 + (int)Math.Floor(a)).ToArray();
            int leftovers = settings.PopulationSize - counts.Sum();
            foreach ( int index in Enumerable.Range( 0, species.Count ).OrderByDescending( i => allocations[i] - Math.Floor( allocations[i] ) ).Take( leftovers ) )
                counts[index]++;

            var nextGeneration = new List<Genome>(settings.PopulationSize);
            for ( int i = 0; i < species.Count; i++ ) {
                Species group = species[i];
                var parents = group.members.OrderByDescending(g => g.fitness).Take(Math.Max(1, (int)Math.Ceiling(group.members.Count * settings.SurvivalFraction))).ToList();
                Genome elite = parents[0].Clone();
                elite.fitness = 0;
                nextGeneration.Add( elite );
                for ( int offspring = 1; offspring < counts[i]; offspring++ ) {
                    Genome parent = SelectParent(parents);
                    Genome child;
                    if ( parents.Count > 1 && RNG.random.NextDouble() < settings.CrossoverChance ) {
                        Genome other = SelectParent(parents);
                        child = parent.fitness >= other.fitness ? Genome.Cross( parent, other ) : Genome.Cross( other, parent );
                    }
                    else {
                        child = parent.Clone();
                    }
                    child.fitness = 0;
                    child.Mutate( settings );
                    nextGeneration.Add( child );
                }
            }
            genomes = nextGeneration;
            Speciate();
            Generation++;
        }

        /*
        ==========================
        ==========================
         */
        private static Genome SelectParent( List<Genome> parents ) {
            double maxFitness = parents[0].fitness;
            if ( maxFitness == 0 ) return parents[RNG.random.Next( parents.Count )];
            double target = RNG.random.NextDouble() * parents.Sum(g => g.fitness / maxFitness);
            foreach ( Genome parent in parents ) {
                target -= parent.fitness / maxFitness;
                if ( target < 0 ) return parent;
            }
            return parents[^1];
        }
    }
}