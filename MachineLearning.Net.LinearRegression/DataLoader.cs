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
using System.IO;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace MachineLearning.Net.LinearRegression
{
    public class DataLoader
    {
        

        public TrainingSet LoadTrainingSetFromFile(string filePath)
        {
            TrainingSet trainingSet = null;
            FileInfo fileInfo= new FileInfo(filePath);
            if(fileInfo.Exists)
            {
                string [] lines=File.ReadAllLines(fileInfo.FullName);
                trainingSet = BuildTrainingSet(lines);                
            }
            return trainingSet;
        }



        public TrainingSet BuildTrainingSet(string[] lines)
        {
            TrainingSet trainingSet = new TrainingSet();
            trainingSet.RowCount = lines.Length;
            trainingSet.ColumnCount = GetColumnCount(lines[0]);
            trainingSet.X = DenseMatrix.Create(trainingSet.RowCount, trainingSet.ColumnCount, 0);
            trainingSet.Y = Vector.Build.Random(lines.Length);
            List<string> features = new List<string>();
            for (int i = 0; i < trainingSet.RowCount; i++)
            {
                features = GetFeaturesInLine(lines[i]);
                trainingSet.X.At(i, 0, 1); // to add the bias feature
                for (int k = 0; k < trainingSet.ColumnCount-1; k++)
                {
                    trainingSet.X.At(i, k+1, double.Parse(features[k]));                    
                }
                trainingSet.Y.At(i, double.Parse(features[trainingSet.ColumnCount-1]));
            }
            return trainingSet;
        }

        private int GetColumnCount(string line)
        {
            int columnCount = GetFeaturesInLine(line).Count;
            return columnCount;
        }

        private List<string> GetFeaturesInLine(string line)
        {
            List<string> features= new List<string>();
            int lineLength=line.Length;
            while(line.IndexOf(",")>0)
            {
                features.Add(line.Substring(0, line.IndexOf(",")));
                line = line.Remove(0, line.IndexOf(",")+1);
            }
            features.Add(line);
            return features;
        }

        

       

     
        
    }
}
