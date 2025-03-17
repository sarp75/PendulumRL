using System;

namespace PendulumRL.Models
{
    public class PendulumCart
    {
        // Cart parameters
        public double CartMass { get; set; } = 1.0; // Mass of the cart
        public double CartPosition { get; set; } // Horizontal position of the cart
        public double CartVelocity { get; set; } // Horizontal velocity of the cart
        public double CartFriction { get; set; } = 0.1; // Friction of the cart
        public double CartPositionMin { get; set; } = -2.0; // Left bound of the track
        public double CartPositionMax { get; set; } = 2.0; // Right bound of the track

        // Pendulum parameters
        public double PendulumLength { get; set; } = 1.0; // Length of the pendulum
        public double PendulumMass { get; set; } = 0.5; // Mass of the pendulum bob
        public double Gravity { get; set; } = 9.81; // Acceleration due to gravity
        public double PendulumDamping { get; set; } = 0.1; // Damping factor for pendulum

        // Pendulum state variables
        public double PendulumAngle { get; private set; } // Angle (0 is hanging down, π is upright)
        public double PendulumAngularVelocity { get; private set; } // Angular velocity

        // For visualization
        public double PendulumX => CartPosition + PendulumLength * Math.Sin(PendulumAngle);
        public double PendulumY => PendulumLength * Math.Cos(PendulumAngle);

        private readonly Random random = new();

        // Constants for angle references
        public const double AngleUpright = 0;

        public PendulumCart(double initialAngle = 0, double initialCartPos = 0.0)
        {
            Reset(initialAngle, initialCartPos);
        }

        public void Reset(double initialAngle = 0.0, double initialCartPos = 0.0)
        {
            // Add some randomness to make learning more robust
            PendulumAngle = initialAngle + (random.NextDouble() - 0.5) * 0.2;
            PendulumAngularVelocity = 0;
            CartPosition = initialCartPos + (random.NextDouble() - 0.5) * 0.2;
            CartVelocity = 0;
        }

        public void Update(double force, double timeStep)
        {
            // Equations of motion for cart-pendulum system
            double totalMass = CartMass + PendulumMass;
            double cosAngle = Math.Cos(PendulumAngle);
            double sinAngle = Math.Sin(PendulumAngle);

            // Calculate accelerations using the dynamics equations for the cart-pendulum system
            double temp =
                (
                    force
                    + PendulumMass
                        * PendulumLength
                        * PendulumAngularVelocity
                        * PendulumAngularVelocity
                        * sinAngle
                    - CartFriction * CartVelocity
                ) / totalMass;

            double pendulumAcceleration =
                (Gravity * sinAngle - cosAngle * temp)
                / (PendulumLength * (4.0 / 3.0 - PendulumMass * cosAngle * cosAngle / totalMass));

            double cartAcceleration =
                temp - PendulumMass * PendulumLength * pendulumAcceleration * cosAngle / totalMass;

            // Apply pendulum damping
            pendulumAcceleration -= PendulumDamping * PendulumAngularVelocity;

            // Update velocities and positions using Euler integration
            PendulumAngularVelocity += pendulumAcceleration * timeStep;
            PendulumAngle += PendulumAngularVelocity * timeStep;

            CartVelocity += cartAcceleration * timeStep;
            CartPosition += CartVelocity * timeStep;

            // Constrain the cart to the track
            if (CartPosition < CartPositionMin)
            {
                CartPosition = CartPositionMin;
                CartVelocity = 0;
            }
            else if (CartPosition > CartPositionMax)
            {
                CartPosition = CartPositionMax;
                CartVelocity = 0;
            }

            // Normalize angle to keep it within -π to π
            while (PendulumAngle > Math.PI)
                PendulumAngle -= 2 * Math.PI;
            while (PendulumAngle < -Math.PI)
                PendulumAngle += 2 * Math.PI;
        }

        public double[] GetState()
        {
            // Return the state as an array:
            // [sin(angle), cos(angle), angular velocity, cart position, cart velocity]
            return
            [
                Math.Sin(PendulumAngle),
                Math.Cos(PendulumAngle),
                PendulumAngularVelocity / 5.0, // Normalize angular velocity
                CartPosition / CartPositionMax, // Normalize position to [-1, 1]
                CartVelocity / 5.0, // Normalize velocity
            ];
        }

        public bool IsBalanced()
        {
            // Check if the pendulum is balanced (close to upright position)
            // Upright is π radians (upside-down position)
            double angleDiff = Math.Abs(PendulumAngle - AngleUpright);
            if (angleDiff > Math.PI)
                angleDiff = 2 * Math.PI - angleDiff;

            // Is balanced when:
            // 1. Angle is near upright (within ~11 degrees)
            // 2. Not rotating too fast
            // 3. Cart not moving too fast
            return angleDiff < 0.2
                && Math.Abs(PendulumAngularVelocity) < 0.5
                && Math.Abs(CartVelocity) < 0.5;
        }

        /// <summary>
        /// Returns a reward value between 0 and 1 based on how upright the pendulum is
        /// </summary>
        public double GetUprightReward()
        {
            // Calculate how close the pendulum is to being upright
            double angleDiff = Math.Abs(PendulumAngle - AngleUpright);
            if (angleDiff > Math.PI)
                angleDiff = 2 * Math.PI - angleDiff;

            // Reward increases as the pendulum gets closer to upright
            // 1.0 when perfectly upright, 0.0 when perfectly hanging down
            double angleReward = 1.0 - (angleDiff / Math.PI);

            // Add penalty for being far from center
            double positionPenalty = Math.Abs(CartPosition / CartPositionMax);

            // Final reward calculation
            return angleReward * (1.0 - 0.5 * positionPenalty);
        }
    }
}
