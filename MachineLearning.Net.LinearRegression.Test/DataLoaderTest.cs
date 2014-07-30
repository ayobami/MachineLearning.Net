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

namespace MachineLearning.Net.LinearRegression.Test
{
    [TestFixture]
    public class DataLoaderTest
    {
        private DataLoader dataLoader;


        [TestFixtureSetUp]
        public void SetupTest()
        {
            dataLoader = new DataLoader();
        }


        [Test]
        public void TestLoadTrainingSetFromFile()
        {
            string path = @"C:\Code Files\ex1data1.txt";
            var trainingSet = dataLoader.LoadTrainingSetFromFile(path);
            Assert.IsNotNull(trainingSet);
        }


        [Test]
        public void TestBuildTrainingSet()
        {
            string [] lines= {"12,14,14", "154,688,477", "78,0.54,0.33", "45,45,2.22233"};
            var trainingSet = dataLoader.BuildTrainingSet(lines);
            Assert.IsNotNull(trainingSet);
        }

        [TestFixtureTearDown]
        public void TearDownTest()
        {
            dataLoader = null;
        }

    }
}
