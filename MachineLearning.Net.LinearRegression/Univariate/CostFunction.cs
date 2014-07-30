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

namespace MachineLearning.Net.LinearRegression.Univariate
{
    public class CostFunction
    {
        public double ComputeCost(TrainingSet trainingSet)
        {
            int m = trainingSet.RowCount;
            Vector<double> theta = Vector<double>.Build.Random(2);
            theta.At(0, 0);
            theta.At(1, 0);
            Matrix<double> hx = trainingSet.X.Multiply(theta.ToColumnMatrix());
            Matrix<double> diff = hx - trainingSet.Y.ToColumnMatrix();           
            Matrix<double> mult = diff.PointwisePower(2);
            double J = (1.0 / (2.0 * m)) * (mult.ColumnSums().Sum());
            return J;
        }

        public double ComputeCost(TrainingSet trainingSet, Vector<double> theta)
        {
            int m = trainingSet.RowCount;
            Matrix<double> hx = trainingSet.X.Multiply(theta.ToColumnMatrix());
            Matrix<double> diff = hx - trainingSet.Y.ToColumnMatrix();
            Matrix<double> mult = diff.PointwisePower(2);
            double J = (1.0 / (2.0 * m)) * (mult.ColumnSums().Sum());
            return J;
        }
    }
}
