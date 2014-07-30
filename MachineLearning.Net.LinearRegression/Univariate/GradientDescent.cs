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
    public class GradientDescent
    {

        public Vector<double> ComputeGradientDescent(TrainingSet trainingSet, Vector<double> theta, double alpha, int iterations)
        {
            int m = trainingSet.RowCount;
            Vector<double> newTheta = Vector<double>.Build.Random(2);
            Matrix<double> hx=null;
            Matrix<double> diff=null;
            Matrix<double> X1 = null;
            Matrix<double> X2 = null;
            Vector<double> Jhistory = Vector<double>.Build.Random(iterations);
            double temp0=0;
            double temp1=0;
            CostFunction costFunction = new CostFunction();
            for (int i = 0; i < iterations; i++)
            {
                 hx = trainingSet.X.Multiply(theta.ToColumnMatrix());
                 diff = hx - trainingSet.Y.ToColumnMatrix();
                 X1 = diff.PointwiseMultiply(trainingSet.X.Column(0).ToColumnMatrix());
                 X2 = diff.PointwiseMultiply(trainingSet.X.Column(1).ToColumnMatrix());
                 temp0=theta.At(0) - alpha*((1.0/m) * X1.ColumnSums().Sum());
                 temp1=theta.At(1) - alpha*((1.0/m) * X2.ColumnSums().Sum());
                 newTheta.At(0, temp0);
                 newTheta.At(1, temp1);
                 double J=costFunction.ComputeCost(trainingSet, newTheta);
                 Jhistory.At(i, J);
                 theta = newTheta;
            }
            return newTheta;
        }
    }
}
