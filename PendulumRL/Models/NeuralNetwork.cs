using System;
using System.Collections.Generic;
using System.Linq;

namespace PendulumRL.Models
{
    public class NeuralNetwork
    {
        // Network structure
        private int[] layers;
        private double[][] neurons;
        private double[][][] weights;
        private double[][] biases;

        // Connection innovation tracking
        private List<Connection> connections = [];
        private List<Node> nodes = [];
        private int nextInnovationId;
        private int nextNodeId;

        // Public access for visualization
        public double[][] Neurons => neurons;
        public double[][][] Weights => weights;
        public int[] Layers => layers;

        // For tracking neural network complexity
        public int ConnectionCount => connections.Count;
        public int NodeCount => nodes.Count;

        private readonly Random random = new();

        // Constructor with layer sizes (e.g., 3, 8, 1 for 3 inputs, 8 hidden, 1 output)
        public NeuralNetwork(params int[] layers)
        {
            this.layers = layers;

            InitializeNeurons(layers);
            InitializeWeightsAndBiases(layers);
            InitializeConnections();
        }

        private void InitializeNeurons(int[] localLayers)
        {
            // Initialize neurons (including input and output layers)
            neurons = new double[localLayers.Length][];
            for (int i = 0; i < localLayers.Length; i++)
            {
                neurons[i] = new double[localLayers[i]];

                // Initialize node objects for tracking
                for (int j = 0; j < localLayers[i]; j++)
                {
                    nodes.Add(
                        new Node
                        {
                            Id = nextNodeId++,
                            Layer = i,
                            Index = j,
                        }
                    );
                }
            }
        }

        private void InitializeWeightsAndBiases(int[] localLayers)
        {
            // Initialize weights and biases
            weights = new double[localLayers.Length - 1][][];
            biases = new double[localLayers.Length - 1][];

            for (int i = 0; i < localLayers.Length - 1; i++)
            {
                biases[i] = new double[localLayers[i + 1]];
                weights[i] = new double[localLayers[i]][];

                for (int j = 0; j < localLayers[i]; j++)
                {
                    weights[i][j] = new double[localLayers[i + 1]];
                    for (int k = 0; k < localLayers[i + 1]; k++)
                    {
                        // Xavier initialization for better learning
                        double scale = Math.Sqrt(6.0 / (localLayers[i] + localLayers[i + 1]));
                        weights[i][j][k] = (random.NextDouble() * 2 - 1) * scale;
                    }
                }
            }
        }

        private void InitializeConnections()
        {
            // Create initial connection objects for all weights
            connections.Clear();

            for (int layer = 0; layer < layers.Length - 1; layer++)
            {
                for (int fromNeuron = 0; fromNeuron < layers[layer]; fromNeuron++)
                {
                    for (int toNeuron = 0; toNeuron < layers[layer + 1]; toNeuron++)
                    {
                        // Find source and target node objects
                        Node sourceNode = nodes.First(n =>
                            n.Layer == layer && n.Index == fromNeuron
                        );
                        Node targetNode = nodes.First(n =>
                            n.Layer == layer + 1 && n.Index == toNeuron
                        );

                        connections.Add(
                            new Connection
                            {
                                InnovationId = nextInnovationId++,
                                FromNode = sourceNode.Id,
                                ToNode = targetNode.Id,
                                Weight = weights[layer][fromNeuron][toNeuron],
                                Enabled = true,
                                LayerConnection = true, // Standard connection between adjacent layers
                                FromLayer = layer,
                                FromIndex = fromNeuron,
                                ToLayer = layer + 1,
                                ToIndex = toNeuron,
                            }
                        );
                    }
                }
            }
        }

        public double[] FeedForward(double[] inputs)
        {
            // Set input layer
            for (int i = 0; i < inputs.Length && i < neurons[0].Length; i++)
            {
                neurons[0][i] = inputs[i];
            }

            // Process each layer
            for (int i = 1; i < layers.Length; i++)
            {
                // Clear the current layer values
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    neurons[i][j] = biases[i - 1][j];
                }

                // Apply standard layer connections
                for (int j = 0; j < neurons[i - 1].Length; j++)
                {
                    for (int k = 0; k < neurons[i].Length; k++)
                    {
                        neurons[i][k] += neurons[i - 1][j] * weights[i - 1][j][k];
                    }
                }

                // Apply skip connections (safely)
                foreach (var conn in connections.Where(c => !c.LayerConnection && c.Enabled))
                {
                    // Make sure indices are valid
                    if (
                        conn.FromLayer >= 0
                        && conn.FromLayer < neurons.Length
                        && conn.FromIndex >= 0
                        && conn.FromIndex < neurons[conn.FromLayer].Length
                        && conn.ToLayer == i
                        && conn.ToIndex >= 0
                        && conn.ToIndex < neurons[i].Length
                    )
                    {
                        neurons[i][conn.ToIndex] +=
                            neurons[conn.FromLayer][conn.FromIndex] * conn.Weight;
                    }
                }

                // Apply activation function
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    neurons[i][j] =
                        i < layers.Length - 1 ? Math.Tanh(neurons[i][j]) : neurons[i][j];
                }
            }

            // Return output layer
            return neurons[layers.Length - 1];
        }

        public void Mutate(double rate = 0.1)
        {
            // Randomly adjust weights for existing connections
            foreach (var conn in connections)
            {
                if (random.NextDouble() < rate)
                {
                    conn.Weight += (random.NextDouble() * 2 - 1) * 0.5; // Adjust by +/- 0.5

                    // Update the weight in the weights array if it's a standard connection
                    if (conn.LayerConnection)
                    {
                        weights[conn.FromLayer][conn.FromIndex][conn.ToIndex] = conn.Weight;
                    }
                }

                // Small chance to toggle connection
                if (random.NextDouble() < rate * 0.1)
                {
                    conn.Enabled = !conn.Enabled;
                }
            }

            // Adjust biases
            for (int i = 0; i < biases.Length; i++)
            {
                for (int j = 0; j < biases[i].Length; j++)
                {
                    if (random.NextDouble() < rate)
                    {
                        biases[i][j] += (random.NextDouble() * 2 - 1) * 0.5; // Adjust by +/- 0.5
                    }
                }
            }

            // Chance to add a new node (split an existing connection)
            if (random.NextDouble() < rate * 0.05 && connections.Any(c => c.Enabled))
            {
                AddNode();
            }

            // Chance to add a new connection between existing nodes
            if (random.NextDouble() < rate * 0.1)
            {
                AddConnection();
            }
        }

        private void AddNode()
        {
            try
            {
                // Select a random enabled connection to split
                var enabledConnections = connections.Where(c => c.Enabled).ToList();
                if (enabledConnections.Count == 0)
                    return;

                var conn = enabledConnections[random.Next(enabledConnections.Count)];

                // Disable the original connection
                conn.Enabled = false;

                // Find the nodes involved in the connection
                Node fromNode = nodes.FirstOrDefault(n => n.Id == conn.FromNode);
                Node toNode = nodes.FirstOrDefault(n => n.Id == conn.ToNode);

                if (fromNode == null || toNode == null)
                {
                    Console.WriteLine("Bad node");
                    return; // Safety check
                }

                // Check if we're adding a new output node
                // Let's assume your output layer is the last layer in the network
                if (fromNode.Layer == layers.Length - 1 || toNode.Layer == layers.Length - 1)
                {
                    // If it's from or to the output layer, suppress the addition of new output nodes
                    Console.WriteLine("Suppressed the addition of new output node.");
                    return; // Skip adding a new node if it's related to the output layer
                }

                // For a standard connection between adjacent layers, we simply add a node in the to-layer
                // For skip connections, we need a different approach
                int newLayerValue;
                int newIndex;

                if (conn.LayerConnection)
                {
                    // Standard connection between adjacent layers
                    newLayerValue = conn.FromLayer + 1;

                    // If this is actually a standard connection between adjacent layers
                    if (newLayerValue == conn.ToLayer)
                    {
                        // Add a new node to the existing layer
                        newIndex = neurons[newLayerValue].Length;

                        // Expand the neurons array for this layer
                        double[] newLayerNeurons = new double[newIndex + 1];
                        Array.Copy(
                            neurons[newLayerValue],
                            newLayerNeurons,
                            neurons[newLayerValue].Length
                        );
                        neurons[newLayerValue] = newLayerNeurons;

                        // Expand the weights arrays
                        // Input weights to the new node
                        double[][] newInputWeights = new double[
                            weights[newLayerValue - 1].Length
                        ][];
                        for (int i = 0; i < weights[newLayerValue - 1].Length; i++)
                        {
                            newInputWeights[i] = new double[neurons[newLayerValue].Length];
                            Array.Copy(
                                weights[newLayerValue - 1][i],
                                newInputWeights[i],
                                weights[newLayerValue - 1][i].Length
                            );
                        }

                        weights[newLayerValue - 1] = newInputWeights;

                        // Output weights from the new node if not the output layer
                        if (newLayerValue < layers.Length - 1)
                        {
                            double[][] newOutputWeights = new double[
                                neurons[newLayerValue].Length
                            ][];
                            for (int i = 0; i < neurons[newLayerValue].Length - 1; i++)
                            {
                                newOutputWeights[i] = weights[newLayerValue][i];
                            }

                            newOutputWeights[neurons[newLayerValue].Length - 1] = new double[
                                neurons[newLayerValue + 1].Length
                            ];
                            weights[newLayerValue] = newOutputWeights;
                        }

                        // Expand the biases array for the new node
                        double[] newBiases = new double[biases[newLayerValue - 1].Length + 1];
                        Array.Copy(
                            biases[newLayerValue - 1],
                            newBiases,
                            biases[newLayerValue - 1].Length
                        );
                        biases[newLayerValue - 1] = newBiases;

                        // Update the layers array
                        layers[newLayerValue]++;
                    }
                    else
                    {
                        // This is a skip connection - treat it as such
                        newLayerValue = conn.ToLayer;
                        newIndex = neurons[newLayerValue].Length;

                        // Similar expansion to above for the target layer
                        double[] newLayerNeurons = new double[newIndex + 1];
                        Array.Copy(
                            neurons[newLayerValue],
                            newLayerNeurons,
                            neurons[newLayerValue].Length
                        );
                        neurons[newLayerValue] = newLayerNeurons;

                        // Update appropriate weight arrays
                        if (newLayerValue > 0)
                        {
                            double[][] newInputWeights = new double[
                                weights[newLayerValue - 1].Length
                            ][];
                            for (int i = 0; i < weights[newLayerValue - 1].Length; i++)
                            {
                                newInputWeights[i] = new double[neurons[newLayerValue].Length];
                                Array.Copy(
                                    weights[newLayerValue - 1][i],
                                    newInputWeights[i],
                                    weights[newLayerValue - 1][i].Length
                                );
                            }

                            weights[newLayerValue - 1] = newInputWeights;
                        }

                        // Expand biases
                        double[] newBiases = new double[biases[newLayerValue - 1].Length + 1];
                        Array.Copy(
                            biases[newLayerValue - 1],
                            newBiases,
                            biases[newLayerValue - 1].Length
                        );
                        biases[newLayerValue - 1] = newBiases;

                        // Update layers array
                        layers[newLayerValue]++;
                    }
                }
                else
                {
                    // Skip connection - place node in one of the existing layers
                    // For simplicity, we'll place it in the "to" layer
                    newLayerValue = conn.ToLayer;
                    newIndex = neurons[newLayerValue].Length;

                    // Similar expansion as above
                    double[] newLayerNeurons = new double[newIndex + 1];
                    Array.Copy(
                        neurons[newLayerValue],
                        newLayerNeurons,
                        neurons[newLayerValue].Length
                    );
                    neurons[newLayerValue] = newLayerNeurons;

                    // Update appropriate weight arrays
                    if (newLayerValue > 0)
                    {
                        double[][] newInputWeights = new double[
                            weights[newLayerValue - 1].Length
                        ][];
                        for (int i = 0; i < weights[newLayerValue - 1].Length; i++)
                        {
                            newInputWeights[i] = new double[neurons[newLayerValue].Length];
                            Array.Copy(
                                weights[newLayerValue - 1][i],
                                newInputWeights[i],
                                weights[newLayerValue - 1][i].Length
                            );
                        }

                        weights[newLayerValue - 1] = newInputWeights;
                    }

                    if (newLayerValue < layers.Length - 1)
                    {
                        double[][] newOutputWeights = new double[neurons[newLayerValue].Length][];
                        for (int i = 0; i < neurons[newLayerValue].Length - 1; i++)
                        {
                            newOutputWeights[i] = weights[newLayerValue][i];
                        }

                        newOutputWeights[neurons[newLayerValue].Length - 1] = new double[
                            neurons[newLayerValue + 1].Length
                        ];
                        weights[newLayerValue] = newOutputWeights;
                    }

                    // Update biases
                    double[] newBiases = new double[biases[newLayerValue - 1].Length + 1];
                    Array.Copy(
                        biases[newLayerValue - 1],
                        newBiases,
                        biases[newLayerValue - 1].Length
                    );
                    biases[newLayerValue - 1] = newBiases;

                    // Update layers array
                    layers[newLayerValue]++;
                }

                // Create the new node
                Node newNode = new Node
                {
                    Id = nextNodeId++,
                    Layer = newLayerValue,
                    Index = newIndex,
                };
                nodes.Add(newNode);

                // Add two new connections: input to new node and new node to output
                connections.Add(
                    new Connection
                    {
                        InnovationId = nextInnovationId++,
                        FromNode = conn.FromNode,
                        ToNode = newNode.Id,
                        Weight = 1.0, // Input to new node has weight 1
                        Enabled = true,
                        LayerConnection = fromNode.Layer + 1 == newNode.Layer,
                        FromLayer = fromNode.Layer,
                        FromIndex = fromNode.Index,
                        ToLayer = newNode.Layer,
                        ToIndex = newNode.Index,
                    }
                );

                connections.Add(
                    new Connection
                    {
                        InnovationId = nextInnovationId++,
                        FromNode = newNode.Id,
                        ToNode = conn.ToNode,
                        Weight = conn.Weight, // Keep original weight
                        Enabled = true,
                        LayerConnection = newNode.Layer + 1 == toNode.Layer,
                        FromLayer = newNode.Layer,
                        FromIndex = newNode.Index,
                        ToLayer = toNode.Layer,
                        ToIndex = toNode.Index,
                    }
                );

                // If these are standard connections, update the weights arrays
                if (fromNode.Layer + 1 == newNode.Layer)
                {
                    weights[fromNode.Layer][fromNode.Index][newNode.Index] = 1.0;
                }

                if (newNode.Layer + 1 == toNode.Layer)
                {
                    weights[newNode.Layer][newNode.Index][toNode.Index] = conn.Weight;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in AddNode: {ex.Message}");
                // Just ignore node addition if it fails
            }
        }

        private void InsertNewLayer(int position)
        {
            // Create a new layer array
            int[] newLayers = new int[layers.Length + 1];
            for (int i = 0; i < position; i++)
            {
                newLayers[i] = layers[i];
            }

            // Add one neuron to the new layer
            newLayers[position] = 1;

            for (int i = position; i < layers.Length; i++)
            {
                newLayers[i + 1] = layers[i];
            }

            // Update layers array
            layers = newLayers;

            // Create new neurons array
            double[][] newNeurons = new double[layers.Length][];
            for (int i = 0; i < position; i++)
            {
                newNeurons[i] = neurons[i];
            }

            // Add one neuron to the new layer
            newNeurons[position] = new double[1];

            for (int i = position; i < neurons.Length; i++)
            {
                newNeurons[i + 1] = neurons[i];
            }

            neurons = newNeurons;

            // Update weights and biases
            double[][][] newWeights = new double[layers.Length - 1][][];
            double[][] newBiases = new double[layers.Length - 1][];

            // Copy weights and biases for layers before the new one
            for (int i = 0; i < position - 1; i++)
            {
                newWeights[i] = weights[i];
                newBiases[i] = biases[i];
            }

            // Create weights for connections to new layer
            newWeights[position - 1] = new double[layers[position - 1]][];
            for (int i = 0; i < layers[position - 1]; i++)
            {
                newWeights[position - 1][i] = new double[1];
                newWeights[position - 1][i][0] = 0.0; // Initialize to zero
            }

            // Create bias for new layer
            newBiases[position - 1] = new double[1] { 0.0 };

            // Create weights for connections from new layer
            newWeights[position] = new double[1][];
            newWeights[position][0] = new double[layers[position + 1]];
            for (int i = 0; i < layers[position + 1]; i++)
            {
                newWeights[position][0][i] = 0.0; // Initialize to zero
            }

            // Copy remaining weights and biases
            for (int i = position; i < weights.Length; i++)
            {
                newWeights[i + 1] = weights[i];
                newBiases[i + 1] = biases[i];
            }

            weights = newWeights;
            biases = newBiases;

            // Update the layer values for all nodes
            foreach (var node in nodes.Where(n => n.Layer >= position))
            {
                node.Layer++;
            }
        }

        private void AddConnection()
        {
            // Try a limited number of times to find valid nodes to connect
            for (int attempts = 0; attempts < 20; attempts++)
            {
                // Pick two random nodes
                Node fromNode = nodes[random.Next(nodes.Count)];
                Node toNode = nodes[random.Next(nodes.Count)];

                // Ensure from node is in an earlier layer than to node
                if (fromNode.Layer >= toNode.Layer)
                {
                    continue;
                }

                // Check if connection already exists
                if (connections.Any(c => c.FromNode == fromNode.Id && c.ToNode == toNode.Id))
                {
                    continue;
                }

                // Create a new connection
                double weight = (random.NextDouble() * 2 - 1) * 2; // Random weight between -2 and 2

                // Is this a standard layer-to-next-layer connection?
                bool isLayerConnection = fromNode.Layer + 1 == toNode.Layer;

                connections.Add(
                    new Connection
                    {
                        InnovationId = nextInnovationId++,
                        FromNode = fromNode.Id,
                        ToNode = toNode.Id,
                        Weight = weight,
                        Enabled = true,
                        LayerConnection = isLayerConnection,
                        FromLayer = fromNode.Layer,
                        FromIndex = fromNode.Index,
                        ToLayer = toNode.Layer,
                        ToIndex = toNode.Index,
                    }
                );

                // If this is a standard connection, update the weight matrix too
                if (isLayerConnection)
                {
                    weights[fromNode.Layer][fromNode.Index][toNode.Index] = weight;
                }

                return; // Successfully added a connection
            }
        }

        public NeuralNetwork Clone()
        {
            // Create a network with the same structure
            var clone = new NeuralNetwork(layers);

            // Copy weights and biases
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        clone.weights[i][j][k] = weights[i][j][k];
                    }
                }
            }

            for (int i = 0; i < biases.Length; i++)
            {
                for (int j = 0; j < biases[i].Length; j++)
                {
                    clone.biases[i][j] = biases[i][j];
                }
            }

            // Copy connection innovations
            clone.connections.Clear();
            foreach (var conn in connections)
            {
                clone.connections.Add(
                    new Connection
                    {
                        InnovationId = conn.InnovationId,
                        FromNode = conn.FromNode,
                        ToNode = conn.ToNode,
                        Weight = conn.Weight,
                        Enabled = conn.Enabled,
                        LayerConnection = conn.LayerConnection,
                        FromLayer = conn.FromLayer,
                        FromIndex = conn.FromIndex,
                        ToLayer = conn.ToLayer,
                        ToIndex = conn.ToIndex,
                    }
                );
            }

            // Copy nodes
            clone.nodes.Clear();
            foreach (var node in nodes)
            {
                clone.nodes.Add(
                    new Node
                    {
                        Id = node.Id,
                        Layer = node.Layer,
                        Index = node.Index,
                    }
                );
            }

            // Set tracking IDs
            clone.nextInnovationId = nextInnovationId;
            clone.nextNodeId = nextNodeId;

            return clone;
        }

        public double CompatibilityDistance(NeuralNetwork other)
        {
            // Calculate compatibility distance between two networks
            // Used for speciation in NEAT

            double c1 = 1.0; // Weight for excess genes
            double c2 = 1.0; // Weight for disjoint genes
            double c3 = 0.4; // Weight for average weight differences

            // Count matching, disjoint, and excess genes
            int matching = 0;
            int disjoint = 0;
            int excess = 0;
            double weightDiff = 0.0;

            int maxInnovation1 = connections.Count > 0 ? connections.Max(c => c.InnovationId) : 0;
            int maxInnovation2 =
                other.connections.Count > 0 ? other.connections.Max(c => c.InnovationId) : 0;

            // Find the maximum innovation ID
            int maxInnovation = Math.Max(maxInnovation1, maxInnovation2);

            // Create dictionaries for quick lookups
            var thisGenes = connections.ToDictionary(c => c.InnovationId);
            var otherGenes = other.connections.ToDictionary(c => c.InnovationId);

            // Compare all genes
            for (int i = 0; i <= maxInnovation; i++)
            {
                bool thisHasGene = thisGenes.ContainsKey(i);
                bool otherHasGene = otherGenes.ContainsKey(i);

                if (thisHasGene && otherHasGene)
                {
                    matching++;
                    weightDiff += Math.Abs(thisGenes[i].Weight - otherGenes[i].Weight);
                }
                else if (thisHasGene && !otherHasGene)
                {
                    if (i <= maxInnovation2)
                        disjoint++;
                    else
                        excess++;
                }
                else if (!thisHasGene && otherHasGene)
                {
                    if (i <= maxInnovation1)
                        disjoint++;
                    else
                        excess++;
                }
            }

            // Calculate the average weight difference
            double averageWeightDiff = matching > 0 ? weightDiff / matching : 0;

            // Normalize by size of larger genome
            int n = Math.Max(connections.Count, other.connections.Count);
            if (n < 1)
                n = 1; // Avoid division by zero

            return (c1 * excess / n) + (c2 * disjoint / n) + (c3 * averageWeightDiff);
        }

        // Combine two networks using crossover (sexual reproduction)
        public static NeuralNetwork Crossover(NeuralNetwork parent1, NeuralNetwork parent2)
        {
            try
            {
                // Create child with same layer structure as parent1
                var child = new NeuralNetwork(parent1.layers);

                // Clear the default connections
                child.connections.Clear();

                // Dictionaries for quick lookups
                var parent1Genes = parent1.connections.ToDictionary(c => c.InnovationId);
                var parent2Genes = parent2.connections.ToDictionary(c => c.InnovationId);

                // Inherit connections
                var allInnovationIds = new HashSet<int>();
                foreach (var conn in parent1.connections)
                    allInnovationIds.Add(conn.InnovationId);
                foreach (var conn in parent2.connections)
                    allInnovationIds.Add(conn.InnovationId);

                foreach (int id in allInnovationIds)
                {
                    Connection childGene = null;

                    if (parent1Genes.ContainsKey(id) && parent2Genes.ContainsKey(id))
                    {
                        // Matching gene - randomly choose from either parent
                        childGene =
                            child.random.NextDouble() < 0.5 ? parent1Genes[id] : parent2Genes[id];
                    }
                    else if (parent1Genes.ContainsKey(id))
                    {
                        // Disjoint or excess from parent1
                        childGene = parent1Genes[id];
                    }
                    else if (parent2Genes.ContainsKey(id))
                    {
                        // Disjoint or excess from parent2
                        childGene = parent2Genes[id];
                    }
                    else
                    {
                        // This shouldn't happen, but just in case
                        continue;
                    }

                    // Safety check before adding - ensure the referenced nodes and layers exist
                    if (
                        childGene.FromLayer < 0
                        || childGene.FromLayer >= child.layers.Length
                        || childGene.ToLayer < 0
                        || childGene.ToLayer >= child.layers.Length
                    )
                    {
                        // Skip connections to non-existent layers
                        continue;
                    }

                    if (
                        childGene.FromIndex < 0
                        || childGene.FromIndex >= child.neurons[childGene.FromLayer].Length
                        || childGene.ToIndex < 0
                        || childGene.ToIndex >= child.neurons[childGene.ToLayer].Length
                    )
                    {
                        // Skip connections to non-existent neurons
                        continue;
                    }

                    // Add the connection to the child
                    child.connections.Add(
                        new Connection
                        {
                            InnovationId = childGene.InnovationId,
                            FromNode = childGene.FromNode,
                            ToNode = childGene.ToNode,
                            Weight = childGene.Weight,
                            Enabled = childGene.Enabled,
                            LayerConnection = childGene.LayerConnection,
                            FromLayer = childGene.FromLayer,
                            FromIndex = childGene.FromIndex,
                            ToLayer = childGene.ToLayer,
                            ToIndex = childGene.ToIndex,
                        }
                    );

                    // Update standard connections in weight matrix
                    if (
                        childGene.LayerConnection
                        && childGene.FromLayer + 1 == childGene.ToLayer
                        && childGene.FromLayer < child.weights.Length
                        && childGene.FromIndex < child.weights[childGene.FromLayer].Length
                        && childGene.ToIndex
                            < child.weights[childGene.FromLayer][childGene.FromIndex].Length
                    )
                    {
                        child.weights[childGene.FromLayer][childGene.FromIndex][childGene.ToIndex] =
                            childGene.Weight;
                    }
                }

                // Inherit nodes
                child.nodes.Clear();
                var allNodeIds = new HashSet<int>();
                foreach (var node in parent1.nodes)
                    allNodeIds.Add(node.Id);
                foreach (var node in parent2.nodes)
                    allNodeIds.Add(node.Id);

                foreach (int id in allNodeIds)
                {
                    Node nodeToAdd = null;

                    if (parent1.nodes.Any(n => n.Id == id))
                    {
                        nodeToAdd = parent1.nodes.First(n => n.Id == id);
                    }
                    else if (parent2.nodes.Any(n => n.Id == id))
                    {
                        nodeToAdd = parent2.nodes.First(n => n.Id == id);
                    }
                    else
                    {
                        // This shouldn't happen, but just in case
                        continue;
                    }

                    // Safety check - ensure the referenced layer exists
                    if (nodeToAdd.Layer < 0 || nodeToAdd.Layer >= child.layers.Length)
                    {
                        // Skip nodes in non-existent layers
                        continue;
                    }

                    // Safety check - ensure the referenced index exists
                    if (
                        nodeToAdd.Index < 0
                        || nodeToAdd.Index >= child.neurons[nodeToAdd.Layer].Length
                    )
                    {
                        // Skip nodes with invalid indices
                        continue;
                    }

                    child.nodes.Add(
                        new Node
                        {
                            Id = nodeToAdd.Id,
                            Layer = nodeToAdd.Layer,
                            Index = nodeToAdd.Index,
                        }
                    );
                }

                // Set tracking IDs to the highest values from parents
                child.nextInnovationId = Math.Max(
                    parent1.nextInnovationId,
                    parent2.nextInnovationId
                );
                child.nextNodeId = Math.Max(parent1.nextNodeId, parent2.nextNodeId);

                return child;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in Crossover: {ex.Message}");

                // Fallback to cloning if crossover fails
                return parent1.Clone();
            }
        }

        // Supporting classes for NEAT
        public class Connection
        {
            public int InnovationId { get; set; }
            public int FromNode { get; set; }
            public int ToNode { get; set; }
            public double Weight { get; set; }
            public bool Enabled { get; set; }

            // For easy mapping to the standard weight matrix
            public bool LayerConnection { get; set; }
            public int FromLayer { get; set; }
            public int FromIndex { get; set; }
            public int ToLayer { get; set; }
            public int ToIndex { get; set; }
        }

        public class Node
        {
            public int Id { get; set; }
            public int Layer { get; set; }
            public int Index { get; set; }
        }
    }
}
