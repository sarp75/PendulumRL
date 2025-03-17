using System;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Collections;
using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Media;
using Avalonia.Threading;
using PendulumRL.Models;

namespace PendulumRL
{
    public partial class MainWindow : Window
    {
        /*private Canvas pendulumCanvas;
        private Canvas networkCanvas;
        private TextBlock txtStatus;
        private TextBlock txtGeneration;
        private TextBlock txtScore;*/

        private PendulumCart pendulumCart;
        private RLAgent agent;

        private bool isRunning;
        private DispatcherTimer timer;

        private int stepsBalanced;
        private int trainingEpisodes = 100;

        public MainWindow()
        {
            InitializeComponent();

            pendulumCanvas = this.FindControl<Canvas>("pendulumCanvas");
            networkCanvas = this.FindControl<Canvas>("networkCanvas");
            txtStatus = this.FindControl<TextBlock>("txtStatus");
            txtGeneration = this.FindControl<TextBlock>("txtGeneration");
            txtScore = this.FindControl<TextBlock>("txtScore");

            var btnReset = this.FindControl<Button>("btnReset");
            var btnTrain = this.FindControl<Button>("btnTrain");
            var btnRun = this.FindControl<Button>("btnRun");

            btnReset.Click += OnResetClick;
            btnTrain.Click += OnTrainClick;
            btnRun.Click += OnRunClick;

            InitializeSimulation();

            timer = new DispatcherTimer
            {
                Interval = TimeSpan.FromMilliseconds(20) // 50 fps
            };
            timer.Tick += OnTimerTick;
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }

        private void InitializeSimulation()
        {
            pendulumCart = new PendulumCart();
            // Now we need 5 inputs: sin(angle), cos(angle), angular velocity, cart position, cart velocity
            agent = new RLAgent(inputSize: 5, hiddenSize: 5, outputSize: 1);
            stepsBalanced = 0;

            UpdateUIStatus("System initialized. Ready to train or run simulation.");
            txtGeneration.Text = "0";
            txtScore.Text = "0";

            // Draw initial state
            DrawPendulumCart();
            DrawNeuralNetwork();
        }

        private async void OnTrainClick(object sender, RoutedEventArgs e)
        {
            if (isRunning) return;

            isRunning = true;
            UpdateUIStatus($"Training in progress... ({trainingEpisodes} episodes)");

            await Task.Run(() =>
            {
                agent.Train(pendulumCart, episodes: trainingEpisodes, stepsPerEpisode: 500);
            });

            isRunning = false;
            UpdateUIStatus("Training complete!");
            txtGeneration.Text = agent.Generation.ToString();
            txtScore.Text = agent.BestScore.ToString("F1");

            // Reset pendulum to start a fresh run
            pendulumCart.Reset(Math.PI, 0); // Start upright
            DrawPendulumCart();
            DrawNeuralNetwork();
        }

        private void OnRunClick(object sender, RoutedEventArgs e)
        {
            if (isRunning)
            {
                timer.Stop();
                isRunning = false;
                UpdateUIStatus("Simulation stopped.");
            }
            else
            {
                pendulumCart.Reset(Math.PI, 0); // Start upright with a slight offset
                stepsBalanced = 0;
                isRunning = true;
                timer.Start();
                UpdateUIStatus("Simulation running...");
            }
        }

        private void OnResetClick(object sender, RoutedEventArgs e)
        {
            if (isRunning)
            {
                timer.Stop();
                isRunning = false;
            }

            InitializeSimulation();
        }

        private void OnTimerTick(object sender, EventArgs e)
        {
            if (!isRunning) return;

            // Get the current state and action from the agent
            double[] state = pendulumCart.GetState();
            double action = agent.GetAction(state);

            // Update pendulum-cart physics
            pendulumCart.Update(action, 0.02); // 20ms time step

            // Check if balanced
            if (pendulumCart.IsBalanced())
            {
                stepsBalanced++;
            }
            else
            {
                stepsBalanced = 0;
            }

            // Update UI
            txtScore.Text = stepsBalanced.ToString();
            UpdateUIStatus($"Running... Steps balanced: {stepsBalanced}");

            // Visualize current state
            DrawPendulumCart();
            DrawNeuralNetwork();
        }

        private void DrawPendulumCart()
        {
            pendulumCanvas.Children.Clear();

            // Get canvas dimensions
            double width = pendulumCanvas.Bounds.Width;
            double height = pendulumCanvas.Bounds.Height;

            if (width <= 0 || height <= 0) return; // Skip if canvas not yet sized

            double floorY = height - 40; // Position of the floor
            double scale = height / 4; // Scale for pendulum length

            // Calculate cart position on screen
            double cartWidth = 80;
            double cartHeight = 30;

            // Map cart position from world coordinates to screen coordinates
            double cartPositionRange = pendulumCart.CartPositionMax - pendulumCart.CartPositionMin;
            double screenPositionRange = width - cartWidth;
            double cartScreenX = ((pendulumCart.CartPosition - pendulumCart.CartPositionMin) / cartPositionRange) * screenPositionRange;

            // Draw the track
            var track = new Line
            {
                StartPoint = new Point(0, floorY + cartHeight / 2),
                EndPoint = new Point(width, floorY + cartHeight / 2),
                Stroke = Brushes.Black,
                StrokeThickness = 3
            };
            pendulumCanvas.Children.Add(track);

            // Draw the cart
            var cart = new Rectangle
            {
                Width = cartWidth,
                Height = cartHeight,
                Fill = Brushes.Blue,
                Stroke = Brushes.Black,
                StrokeThickness = 2
            };
            Canvas.SetLeft(cart, cartScreenX);
            Canvas.SetTop(cart, floorY - cartHeight);
            pendulumCanvas.Children.Add(cart);

            // Draw the pivot point on the cart
            double pivotX = cartScreenX + cartWidth / 2;
            double pivotY = floorY - cartHeight;

            var pivot = new Ellipse
            {
                Width = 10,
                Height = 10,
                Fill = Brushes.Black
            };
            Canvas.SetLeft(pivot, pivotX - 5);
            Canvas.SetTop(pivot, pivotY - 5);
            pendulumCanvas.Children.Add(pivot);

            // Calculate pendulum bob position
            double bobX = pivotX + Math.Sin(pendulumCart.PendulumAngle) * scale;
            double bobY = pivotY - Math.Cos(pendulumCart.PendulumAngle) * scale;

            // Draw pendulum rod
            var rod = new Line
            {
                StartPoint = new Point(pivotX, pivotY),
                EndPoint = new Point(bobX, bobY),
                Stroke = Brushes.Black,
                StrokeThickness = 3
            };
            pendulumCanvas.Children.Add(rod);

            // Draw pendulum bob
            var bob = new Ellipse
            {
                Width = 30,
                Height = 30,
                Fill = Brushes.Red,
                Stroke = Brushes.Black,
                StrokeThickness = 2
            };
            Canvas.SetLeft(bob, bobX - 15);
            Canvas.SetTop(bob, bobY - 15);
            pendulumCanvas.Children.Add(bob);

            // Draw upright target position (faded)
            var targetLine = new Line
            {
                StartPoint = new Point(pivotX, pivotY),
                EndPoint = new Point(pivotX, pivotY - scale),
                Stroke = new SolidColorBrush(Color.FromArgb(80, 0, 200, 0)), // Semi-transparent green
                StrokeThickness = 2,
                StrokeDashArray = new AvaloniaList<double> { 4, 2 } // Dashed line
            };
            pendulumCanvas.Children.Add(targetLine);

            // Draw force arrow
            if (agent != null)
            {
                double[] state = pendulumCart.GetState();
                double force = agent.GetAction(state);

                // Only draw if force is significant
                if (Math.Abs(force) > 0.5)
                {
                    double arrowLength = Math.Min(Math.Abs(force) * 5, 50);
                    double arrowX = pivotX + (force > 0 ? arrowLength : -arrowLength);

                    var forceArrow = new Line
                    {
                        StartPoint = new Point(pivotX, pivotY - cartHeight/2),
                        EndPoint = new Point(arrowX, pivotY - cartHeight/2),
                        Stroke = force > 0 ? Brushes.Green : Brushes.Red,
                        StrokeThickness = 4
                    };
                    pendulumCanvas.Children.Add(forceArrow);

                    // Draw arrowhead
                    double headSize = 10;
                    var arrowhead = new Polygon
                    {
                        Points = new AvaloniaList<Point>(),
                        Fill = force > 0 ? Brushes.Green : Brushes.Red
                    };

                    if (force > 0)
                    {
                        arrowhead.Points.Add(new Point(arrowX, pivotY - cartHeight/2));
                        arrowhead.Points.Add(new Point(arrowX - headSize, pivotY - cartHeight/2 - headSize/2));
                        arrowhead.Points.Add(new Point(arrowX - headSize, pivotY - cartHeight/2 + headSize/2));
                    }
                    else
                    {
                        arrowhead.Points.Add(new Point(arrowX, pivotY - cartHeight/2));
                        arrowhead.Points.Add(new Point(arrowX + headSize, pivotY - cartHeight/2 - headSize/2));
                        arrowhead.Points.Add(new Point(arrowX + headSize, pivotY - cartHeight/2 + headSize/2));
                    }

                    pendulumCanvas.Children.Add(arrowhead);
                }
            }

            // Draw center marker
            var centerLine = new Line
            {
                StartPoint = new Point(width / 2, floorY - 10),
                EndPoint = new Point(width / 2, floorY + 10),
                Stroke = Brushes.Gray,
                StrokeThickness = 1
            };
            pendulumCanvas.Children.Add(centerLine);
        }

        private void DrawNeuralNetwork()
        {
            networkCanvas.Children.Clear();

            var network = agent.BestNetwork;
            if (network == null) return;

            double width = networkCanvas.Bounds.Width;
            double height = networkCanvas.Bounds.Height;

            if (width <= 0 || height <= 0) return; // Skip if canvas not yet sized

            var neurons = network.Neurons;
            var weights = network.Weights;

            // Calculate positions for all neurons
            double horizontalSpacing = width / (neurons.Length + 1);
            var neuronPositions = new Point[neurons.Length][];

            for (int layer = 0; layer < neurons.Length; layer++)
            {
                double layerHeight = height / (neurons[layer].Length + 1);
                neuronPositions[layer] = new Point[neurons[layer].Length];

                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    double x = horizontalSpacing * (layer + 1);
                    double y = layerHeight * (neuron + 1);
                    neuronPositions[layer][neuron] = new Point(x, y);
                }
            }

            // Draw connections (weights) first so they're behind neurons
            for (int layer = 0; layer < weights.Length; layer++)
            {
                for (int fromNeuron = 0; fromNeuron < weights[layer].Length; fromNeuron++)
                {
                    Point from = neuronPositions[layer][fromNeuron];

                    for (int toNeuron = 0; toNeuron < weights[layer][fromNeuron].Length; toNeuron++)
                    {
                        Point to = neuronPositions[layer + 1][toNeuron];
                        double weight = weights[layer][fromNeuron][toNeuron];

                        // Determine line thickness and color based on weight
                        double thickness = Math.Clamp(Math.Abs(weight) * 2, 0.5, 4);
                        var brush = weight > 0 ? Brushes.Green : Brushes.Red;

                        // Create line with opacity based on weight magnitude
                        byte alpha = (byte)Math.Clamp(Math.Abs(weight) * 255, 40, 255);
                        var adjustedBrush = new SolidColorBrush(Color.FromArgb(
                            alpha,
                            brush.Color.R,
                            brush.Color.G,
                            brush.Color.B
                        ));

                        var line = new Line
                        {
                            StartPoint = from,
                            EndPoint = to,
                            Stroke = adjustedBrush,
                            StrokeThickness = thickness
                        };

                        networkCanvas.Children.Add(line);
                    }
                }
            }

            // Draw neurons
            for (int layer = 0; layer < neurons.Length; layer++)
            {
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    Point pos = neuronPositions[layer][neuron];
                    double value = neurons[layer][neuron];

                    // Calculate neuron color based on activation
                    byte intensity = (byte)Math.Clamp((value + 1) * 127.5, 0, 255); // Map [-1,1] to [0,255]
                    var brush = new SolidColorBrush(Color.FromRgb(intensity, intensity, 255));

                    // Draw neuron circle
                    double neuronSize = layer == neurons.Length - 1 ? 20 : 15; // Output neurons slightly bigger
                    var circle = new Ellipse
                    {
                        Width = neuronSize,
                        Height = neuronSize,
                        Fill = brush,
                        Stroke = Brushes.Black,
                        StrokeThickness = 1
                    };

                    Canvas.SetLeft(circle, pos.X - neuronSize / 2);
                    Canvas.SetTop(circle, pos.Y - neuronSize / 2);
                    networkCanvas.Children.Add(circle);

                    // Add labels for input and output neurons
                    if (layer == 0 || layer == neurons.Length - 1)
                    {
                        string label = "";
                        if (layer == 0)
                        {
                            switch (neuron)
                            {
                                case 0:
                                    label = "sin(θ)";
                                    break;
                                case 1:
                                    label = "cos(θ)";
                                    break;
                                case 2:
                                    label = "ω";
                                    break;
                                case 3:
                                    label = "pos";
                                    break;
                                case 4:
                                    label = "vel";
                                    break;
                            }
                        }
                        else // Output layer
                        {
                            label = "Force";
                        }

                        var text = new TextBlock
                        {
                            Text = label,
                            FontSize = 12,
                            Foreground = Brushes.Black
                        };

                        // Position text near neuron
                        double textOffsetX = layer == 0 ? -40 : 30; // Before inputs, after outputs

                        Canvas.SetLeft(text, pos.X + textOffsetX);
                        Canvas.SetTop(text, pos.Y - 8); // Center text vertically
                        networkCanvas.Children.Add(text);
                    }

                    // Show activation value inside neuron
                    var valueText = new TextBlock
                    {
                        Text = value.ToString("F1"),
                        FontSize = 10,
                        Foreground = Brushes.Black,
                        HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center,
                        VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center
                    };

                    Canvas.SetLeft(valueText, pos.X - 10);
                    Canvas.SetTop(valueText, pos.Y - 7);
                    networkCanvas.Children.Add(valueText);
                }
            }

            // Draw layer labels
            string[] layerLabels = { "Input", "Hidden", "Output" };
            for (int layer = 0; layer < neurons.Length; layer++)
            {
                var labelText = new TextBlock
                {
                    Text = layerLabels[Math.Min(layer, layerLabels.Length - 1)],
                    FontSize = 14,
                    Foreground = Brushes.Black,
                    FontWeight = FontWeight.Bold
                };

                double x = horizontalSpacing * (layer + 1);
                Canvas.SetLeft(labelText, x - 25); // Center text
                Canvas.SetTop(labelText, 10); // Position at top
                networkCanvas.Children.Add(labelText);
            }
        }

        private void UpdateUIStatus(string message)
        {
            txtStatus.Text = message;
        }
    }
}