using System;
using System.Collections.Generic;
using System.Linq;

namespace PendulumRL.Models
{
    // ReSharper disable once InconsistentNaming
    public class RLAgent
    {
        private readonly int inputSize;
        private readonly int outputSize;
        private readonly int hiddenSize;

        public NeuralNetwork BestNetwork { get; private set; }

        // NEAT specific properties
        private List<NeuralNetwork> population = new List<NeuralNetwork>();
        private List<Species> species = new List<Species>();
        private int populationSize = 20;
        private double speciationThreshold = 3.0;
        private double mutationRate = 0.2;
        private double addNodeProbability = 0.03;
        private double addConnectionProbability = 0.05;

        private double bestScore = double.MinValue;
        private int generation;

        public int Generation => generation;
        public double BestScore => bestScore;

        private readonly Random random = new Random();

        public RLAgent(int inputSize, int hiddenSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            // Initialize population with basic networks
            InitializePopulation();

            // Set the initial best network
            BestNetwork = population[0].Clone();
        }

        private void InitializePopulation()
        {
            population.Clear();

            // Create initial population with the same initial structure
            for (int i = 0; i < populationSize; i++)
            {
                var network = new NeuralNetwork(inputSize, hiddenSize, hiddenSize, hiddenSize, outputSize);

                // Add small mutations to ensure diversity
                if (i > 0)
                {
                    network.Mutate(mutationRate);
                }

                population.Add(network);
            }
        }

        public double GetAction(double[] state)
        {
            double[] output = BestNetwork.FeedForward(state);
            // Map output from [-1, 1] to a reasonable force range
            return output[0] * 10.0; // Scale to +/- 10 force units
        }

        public void Train(PendulumCart pendulum, int episodes = 100, int stepsPerEpisode = 500)
        {
            // Start with a low mutation rate and gradually increase
            double currentMutationRate = 0.05;

            // Run for the specified number of episodes (generations)
            for (int episode = 0; episode < episodes; episode++)
            {
                // Gradually increase mutation rate as training progresses
                if (episode > 10)
                    currentMutationRate = 0.1;
                if (episode > 20)
                    currentMutationRate = 0.15;
                if (episode > 50)
                    currentMutationRate = mutationRate; // Full rate

                // Evaluate each network in the population
                var scores = EvaluatePopulation(pendulum, stepsPerEpisode);

                // Find the best network in this generation
                int bestIndex = 0;
                for (int i = 1; i < population.Count; i++)
                {
                    if (scores[i] > scores[bestIndex])
                    {
                        bestIndex = i;
                    }
                }

                // If new best found, update our best network
                if (scores[bestIndex] > bestScore)
                {
                    bestScore = scores[bestIndex];
                    BestNetwork = population[bestIndex].Clone();
                    generation++;

                    Console.WriteLine(
                        $"Generation {generation}: New best score = {bestScore}, "
                            + $"Nodes: {BestNetwork.NodeCount}, Connections: {BestNetwork.ConnectionCount}"
                    );
                }

                // Perform speciation - group networks into species
                SpeciatePopulation();

                // Reproduce to create the next generation
                try
                {
                    ReproduceAndReplace(scores);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error in reproduction: {ex.Message}");
                    // Simple fallback: just mutate the existing population
                    for (int i = 0; i < population.Count; i++)
                    {
                        if (i != bestIndex) // Don't mutate the best one
                        {
                            population[i].Mutate(currentMutationRate);
                        }
                    }
                }
            }
        }

        private double[] EvaluatePopulation(PendulumCart pendulum, int stepsPerEpisode)
        {
            var scores = new double[population.Count];

            // Evaluate each network in the population
            for (int i = 0; i < population.Count; i++)
            {
                double totalScore = 0;
                // Run multiple trials for better evaluation
                for (int trial = 0; trial < 3; trial++)
                {
                    // Start slightly off from hanging position to encourage learning
                    pendulum.Reset(0.1, 0); // Small angle from hanging down

                    double episodeScore = 0;
                    int balancedSteps = 0;
                    int consecutiveBalanced = 0; // Count consecutive balance steps for bonus

                    for (int step = 0; step < stepsPerEpisode; step++)
                    {
                        double[] state = pendulum.GetState();
                        double[] output = population[i].FeedForward(state);
                        double action = output[0] * 10.0; // Scale to +/- 10 force units

                        pendulum.Update(action, 0.02); // 20ms time step

                        // Get reward based on how upright the pendulum is
                        double reward = pendulum.GetUprightReward();
                        episodeScore += reward;

                        // Add bonus for consecutive balanced steps
                        if (pendulum.IsBalanced())
                        {
                            balancedSteps++;
                            consecutiveBalanced++;

                            // Exponential bonus for staying balanced longer
                            double bonus = Math.Min(0.5, 0.1 * Math.Log10(consecutiveBalanced + 1));
                            episodeScore += bonus;
                        }
                        else
                        {
                            consecutiveBalanced = 0;
                        }

                        // Stop early if pendulum has fallen and cart is near edge
                        // This speeds up evaluation of poor networks
                        if (step > 100 && reward < 0.1 && Math.Abs(pendulum.CartPosition) > 1.9)
                        {
                            // Apply penalty for failing early
                            episodeScore -= 10;
                            break;
                        }
                    }

                    // Bonus for total balanced steps
                    episodeScore += balancedSteps * 0.1;

                    totalScore += episodeScore;
                }

                scores[i] = totalScore / 3.0; // Average score across trials
            }

            return scores;
        }

        private void SpeciatePopulation()
        {
            // Clear existing species
            foreach (var s in species)
            {
                s.Members.Clear();
            }

            // For each genome in the population
            for (int i = 0; i < population.Count; i++)
            {
                bool foundSpecies = false;

                // Try to add to an existing species
                foreach (var s in species)
                {
                    double distance = population[i].CompatibilityDistance(s.Representative);

                    if (distance < speciationThreshold)
                    {
                        s.Members.Add(i);
                        foundSpecies = true;
                        break;
                    }
                }

                // If not compatible with any existing species, create a new one
                if (!foundSpecies)
                {
                    var newSpecies = new Species
                    {
                        Representative = population[i],
                        Members = new List<int> { i },
                    };
                    species.Add(newSpecies);
                }
            }

            // Remove empty species
            species.RemoveAll(s => s.Members.Count == 0);
        }

        private void ReproduceAndReplace(double[] scores)
        {
            try
            {
                var newPopulation = new List<NeuralNetwork>();

                // Calculate adjusted fitness and total adjusted fitness for each species
                foreach (var s in species)
                {
                    s.TotalAdjustedFitness = 0;
                    s.AdjustedFitnesses = new double[s.Members.Count];

                    // Calculate adjusted fitness for each member (fitness / species size)
                    for (int i = 0; i < s.Members.Count; i++)
                    {
                        int memberIndex = s.Members[i];
                        s.AdjustedFitnesses[i] = scores[memberIndex] / s.Members.Count;
                        s.TotalAdjustedFitness += s.AdjustedFitnesses[i];
                    }
                }

                // Calculate total adjusted fitness across all species
                double totalFitness = species.Sum(s => s.TotalAdjustedFitness);

                // Ensure best network is preserved (elitism)
                int overallBestIndex = 0;
                for (int i = 1; i < population.Count; i++)
                {
                    if (scores[i] > scores[overallBestIndex])
                    {
                        overallBestIndex = i;
                    }
                }

                newPopulation.Add(population[overallBestIndex].Clone());

                // Reproduce in each species proportionally to its adjusted fitness
                foreach (var s in species)
                {
                    // Calculate number of offspring for this species based on fitness share
                    int offspring =
                        totalFitness > 0
                            ? (int)
                                Math.Round(
                                    (s.TotalAdjustedFitness / totalFitness) * (populationSize - 1)
                                )
                            : 1; // Give each species at least one offspring if all fitnesses are 0

                    // Generate offspring
                    for (int i = 0; i < offspring; i++)
                    {
                        // Select parent(s)
                        int parent1Index = SelectParentFromSpecies(s);

                        // Sometimes do crossover, sometimes just mutation
                        if (random.NextDouble() < 0.75 && s.Members.Count > 1)
                        {
                            // Crossover
                            int parent2Index = SelectParentFromSpecies(s, parent1Index);

                            NeuralNetwork child = NeuralNetwork.Crossover(
                                population[s.Members[parent1Index]],
                                population[s.Members[parent2Index]]
                            );

                            // Mutate the result
                            child.Mutate(mutationRate);
                            newPopulation.Add(child);
                        }
                        else
                        {
                            // Just clone and mutate
                            var child = population[s.Members[parent1Index]].Clone();
                            child.Mutate(mutationRate);
                            newPopulation.Add(child);
                        }
                    }
                }

                // If we didn't generate enough networks, add more
                while (newPopulation.Count < populationSize)
                {
                    // Create a new one based on the best
                    var extra = population[overallBestIndex].Clone();
                    extra.Mutate(mutationRate * 2); // Higher mutation for diversity
                    newPopulation.Add(extra);
                }

                // If we generated too many networks, trim
                while (newPopulation.Count > populationSize)
                {
                    newPopulation.RemoveAt(newPopulation.Count - 1);
                }

                // Replace the population
                population = newPopulation;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error from ReproduceAndReplace" + ex.Message);
            }
        }

        private int SelectParentFromSpecies(Species species, int? exclude = null)
        {
            // Use tournament selection within the species
            int tournamentSize = Math.Min(3, species.Members.Count);
            int bestIndex = -1;
            double bestFitness = -double.MaxValue;

            for (int i = 0; i < tournamentSize; i++)
            {
                int candidateIndex;
                do
                {
                    candidateIndex = random.Next(species.Members.Count);
                } while (candidateIndex == exclude);

                if (species.AdjustedFitnesses[candidateIndex] > bestFitness)
                {
                    bestIndex = candidateIndex;
                    bestFitness = species.AdjustedFitnesses[candidateIndex];
                }
            }

            return bestIndex;
        }
    }

    // Supporting class for speciation in NEAT
    public class Species
    {
        public NeuralNetwork Representative { get; set; } // Exemplar network for this species
        public List<int> Members { get; set; } = new List<int>(); // Indices of members in population
        public double TotalAdjustedFitness { get; set; } // Total adjusted fitness for the species
        public double[] AdjustedFitnesses { get; set; } // Adjusted fitness for each member
    }
}
