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
    public class GradientDescentTest
    {
        private DataLoader dataLoader;
        private GradientDescent gradientDescent;


        [TestFixtureSetUp]
        public void SetupTest()
        {
            dataLoader = new DataLoader();
            gradientDescent = new GradientDescent();
        }

        [Test]
        public void TestComputeGradientDescent()
        {
            Vector<double> theta = Vector<double>.Build.Random(2);
            theta.At(0, 0);
            theta.At(1, 0);
            string path = @"C:\Code Files\ex1data1.txt";
            var trainingSet = dataLoader.LoadTrainingSetFromFile(path);
            var newTheta=gradientDescent.ComputeGradientDescent(trainingSet, theta, 0.01, 1500);
            Assert.AreEqual(newTheta.At(0), double.Parse("-3.63029143940436"));
            
        }

        [TestFixtureTearDown]
        public void TearDownTest()
        {
            dataLoader = null;
            gradientDescent = null;
        }

    }
}
