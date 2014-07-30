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
    public class CostFunctionTest
    {
        private CostFunction costFunction;
        private DataLoader dataLoader;


        [TestFixtureSetUp]
        public void SetupTest()
        {
            costFunction = new CostFunction();
            dataLoader = new DataLoader();
        }


        [Test]
        public void TestComputeCost()
        {
            string path = @"C:\Code Files\ex1data1.txt";
            var trainingSet = dataLoader.LoadTrainingSetFromFile(path);
            var cost=costFunction.ComputeCost(trainingSet);
            Assert.AreEqual(cost, double.Parse("32.072733877455654"));
        }


        [Test]
        public void TestComputeCost2()
        {
            Vector<double> theta = Vector<double>.Build.Random(2);
            theta.At(0, 0);
            theta.At(1, 0);
            string path = @"C:\Code Files\ex1data1.txt";
            var trainingSet = dataLoader.LoadTrainingSetFromFile(path);
            var cost = costFunction.ComputeCost(trainingSet, theta);
            Assert.AreEqual(cost, double.Parse("32.072733877455654"));
        }

        [TestFixtureTearDown]
        public void TearDownTest()
        {
            costFunction = null;
            dataLoader = null;
        }

    }
}
