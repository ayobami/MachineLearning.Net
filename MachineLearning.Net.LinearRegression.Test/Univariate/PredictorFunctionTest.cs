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
using NUnit.Framework;
using MachineLearning.Net.LinearRegression;
using MachineLearning.Net.LinearRegression.Univariate;
using MathNet.Numerics.LinearAlgebra;

namespace MachineLearning.Net.LinearRegression.Test.Univariate
{
    [TestFixture]
    public class PredictorFunctionTest
    {
        private DataLoader dataLoader;
        private GradientDescent gradientDescent;
        private PredictorFunction predictorFunction;


        [TestFixtureSetUp]
        public void SetupTest()
        {
            dataLoader = new DataLoader();
            gradientDescent = new GradientDescent();
            predictorFunction = new PredictorFunction();
        }

        [Test]
        public void TestPredict()
        {
            Vector<double> theta = Vector<double>.Build.Random(2);
            theta.At(0, 0);
            theta.At(1, 0);
            string path = @"C:\Code Files\ex1data1.txt";
            var trainingSet = dataLoader.LoadTrainingSetFromFile(path);
            var newTheta=gradientDescent.ComputeGradientDescent(trainingSet, theta, 0.01, 1500);
            var prediction=predictorFunction.Predict(3.5, newTheta);
            Assert.AreEqual(prediction, double.Parse("0.45197678677017672"));
           
        }

        [Test]
        public void TestPredict2()
        {
            Vector<double> theta = Vector<double>.Build.Random(2);
            theta.At(0, 0);
            theta.At(1, 0);
            string path = @"C:\Code Files\ex1data1.txt";
            var trainingSet = dataLoader.LoadTrainingSetFromFile(path);
            var newTheta = gradientDescent.ComputeGradientDescent(trainingSet, theta, 0.01, 1500);
            List<double> values = new List<double> { 3.5, 7 };
            var predictions = predictorFunction.Predict(values, newTheta);
            CollectionAssert.IsNotEmpty(predictions);

        }

        [TestFixtureTearDown]
        public void TearDownTest()
        {
            dataLoader = null;
            gradientDescent = null;
            predictorFunction = null;
        }

    }
}
