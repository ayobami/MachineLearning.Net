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
    public class PredictorFunction
    {
        public double Predict(double value,  Vector<double> theta)
        {
            Matrix<double> x = DenseMatrix.Create(1,2,0);
            x.At(0,0,1);
            x.At(0,1,value);
            double prediction = x.Multiply( theta.ToColumnMatrix()).ColumnSums().Sum();
            return prediction;
        }

        public List<double> Predict(List<double> values, Vector<double> theta)
        {
            Matrix<double> x = DenseMatrix.Create(1, 2, 0);
            x.At(0, 0, 1);
            double prediction = 0;
            List<double> predictions = new List<double>();
            for (int i = 0; i < values.Count; i++)
            {
                
                x.At(0, 1, values[i]);
                prediction = x.Multiply(theta.ToColumnMatrix()).ColumnSums().Sum();
                predictions.Add(prediction);
            }
            return predictions;
        }
    }
}
