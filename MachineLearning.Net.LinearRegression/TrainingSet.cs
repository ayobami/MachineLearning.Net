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

namespace MachineLearning.Net.LinearRegression
{
    public class TrainingSet
    {
        public Matrix<double> X {get;set;}

        public Vector<double> Y { get; set; }

        public int RowCount { get; set; }

        public int ColumnCount { get; set; }
    }
}
