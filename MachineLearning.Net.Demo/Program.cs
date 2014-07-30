/*
 * This code was developed 
 *
 *          by
 *
 *     Ayobami Adewole
 * 
 * www.ayobamiadewole.com
 *
 * Licensed under Mozilla Public License, version 2.0 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MachineLearning.Net.LinearRegression;
using MachineLearning.Net.LinearRegression.Univariate;

namespace MachineLearning.Net.Demo
{
    class Program
    {
        static void Main(string[] args)
        {
            LinearRegressionDemo();
            Console.ReadLine();
        }

        public static void LinearRegressionDemo()
        {
            //    Linear Regression Univariate
            //This algorithm is used on training set of Regression problems 
            //with only one feature or variable



            //Loading the training set from file
            string path =Environment.CurrentDirectory+@"\ex1data1.txt";
            DataLoader dataLoader = new DataLoader();
            TrainingSet trainingSet = dataLoader.LoadTrainingSetFromFile(path);

            //theta initialization
            Vector<double> theta = Vector<double>.Build.Random(2);
            theta.At(0, 0);
            theta.At(1, 0);
            int iterations = 1500; // gradient descent iterations
            double alpha = 0.01; // learning rate alpha

            //computing gradient descent
            GradientDescent gradientDescent = new GradientDescent();
            theta = gradientDescent.ComputeGradientDescent(trainingSet, theta, alpha, iterations);

            //predicting
            PredictorFunction predictorFunction = new PredictorFunction();
            double prediction = predictorFunction.Predict(3.5, theta);

            Console.WriteLine("Prediction for value {0} is {1}", 3.5, prediction);
        }
    }
}
