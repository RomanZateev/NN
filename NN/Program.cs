using System;
using System.Collections.Generic;
//тест
namespace NN
{
    class Program
    {
        static void Main(string[] args)
        {
            train();
        }

        class sigmoid
        {
            public static double output(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }

            public static double derivative(double x)
            {
                return x * (1 - x);
            }
        }

        public static int numberOfneurons = 3;

        class Neuron
        {
            public double[] inputs = new double[numberOfneurons];
            public double[] weights = new double[numberOfneurons];
            public double error;

            private double biasWeight;

            private Random r = new Random();

            public double output
            {
                get
                {
                    double output = 0.0;

                    for (int i = 0; i < numberOfneurons; i++)
                        output += inputs[i] * weights[i];

                    return sigmoid.output(output + biasWeight);
                }
            }

            public void randomizeWeights()
            {
                for (int i = 0; i < weights.Length; i++)
                    weights[i] = r.NextDouble();
                biasWeight = r.NextDouble();
            }

            public void adjustWeights()
            {
                for (int i = 0; i < weights.Length; i++)
                    weights[i] += error * inputs[i];
                biasWeight += error;
            }
        }

        private static void train()
        {
            double[,] inputs =
            {
                { 1, 1, 0 },
                { 1, 0, 1 },
                { 0, 1, 1 },
                { 0, 1, 0 },
                { 0, 1, 1 },
                { 0, 0, 1 },
                { 0, 1, 0 },
                { 1, 1, 1 },
                { 0, 0, 0 }
            };

            // desired results
            double[] results = { 1, 1, 2, 3, 2, 3, 3, 1, 2 };

            double max = -1;

            // find max value
            for (int i = 0; i < results.Length; i++)
                if (results[i] > max)
                    max = results[i];

            // normalizing
            for (int i = 0; i < results.Length; i++)
                results[i] /= max;

            // creating the neurons
            List<Neuron> hiddenNeurons = new List<Neuron>(numberOfneurons);

            for (int i = 0; i < numberOfneurons; i++)
                hiddenNeurons.Add(new Neuron());

            Neuron outputNeuron = new Neuron();

            // random weights
            foreach (Neuron hiddenNeuron in hiddenNeurons)
                hiddenNeuron.randomizeWeights();

            outputNeuron.randomizeWeights();

            int epoch = 0;

            Retry:
            epoch++;
            for (int i = 0; i < 9; i++)
            {
                // 1) forward propagation (calculates output)
                outputNeuron.inputs = fillIN(inputs[i, 0], inputs[i, 1], inputs[i, 2], hiddenNeurons);

                Console.WriteLine("{0} and {1} and {2} = {3}", inputs[i, 0], inputs[i, 1], inputs[i, 2], outputNeuron.output * numberOfneurons);

                // 2) back propagation (adjusts weights)

                // adjusts the weight of the output neuron, based on its error
                outputNeuron.error = sigmoid.derivative(outputNeuron.output) * (results[i] - outputNeuron.output);
                outputNeuron.adjustWeights();

                // then adjusts the hidden neurons' weights, based on their errors
                for (int j = 0; j < numberOfneurons; j++)
                    hiddenNeurons[j].error = sigmoid.derivative(hiddenNeurons[j].output) * outputNeuron.error * outputNeuron.weights[j];

                foreach (Neuron hiddenNeuron in hiddenNeurons)
                    hiddenNeuron.adjustWeights();
            }

            if (epoch < 1000)
                goto Retry;

            Console.WriteLine();
            Console.WriteLine("Network have been trained successfully");

            while (true)
            {
                Console.WriteLine("________________________");
                Console.WriteLine("Print your binary values");

                Console.Write("value 1: ");
                int value1 = Convert.ToInt32(Console.ReadLine());

                Console.Write("value 2: ");
                int value2 = Convert.ToInt32(Console.ReadLine());

                Console.Write("value 3: ");
                int value3 = Convert.ToInt32(Console.ReadLine());

                outputNeuron.inputs = fillIN(value1, value2, value3, hiddenNeurons);

                Console.WriteLine("result : " + (outputNeuron.output * max).ToString("0"));
            }
        }

        private static double[] fillIN(double value1, double value2, double value3, List<Neuron> hiddenNeurons)
        {
            foreach (Neuron hiddenNeuron in hiddenNeurons)
                hiddenNeuron.inputs = new double[] { value1, value2, value3 };

            double[] outputNeuronInputs = new double[numberOfneurons];

            for (int j = 0; j < numberOfneurons; j++)
                outputNeuronInputs[j] = hiddenNeurons[j].output;

            return outputNeuronInputs;
        }
    }
}
