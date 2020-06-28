using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace LAT.BinaryClassification
{
    public class BinaryClassifier
    {
                private bool initalized = false;

        private TrainTestData splitDataView;
        private ITransformer model;
        static readonly string _dataPath = "/home/max/Data/sentiment_labelled_sentences/yelp_labelled.txt";

        private readonly MLContext mlContext;

        public BinaryClassifier(MLContext context)
        {
            mlContext = context ?? throw new ArgumentNullException(nameof(context));
        }

        public void Initalize()
        {
            LoadData();
            BuildAndTrainModel();

            initalized = true;
        }

        private void LoadData()
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<BinaryClassData>(_dataPath, hasHeader: false);
            splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            // return splitDataView;
        }

        private void BuildAndTrainModel()
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(BinaryClassData.Value))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Create and Train the Model ===============");
            model = estimator.Fit(splitDataView.TrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
        }

        public void Evaluate()
        {
            if (initalized)
            {
                Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
                IDataView predictions = model.Transform(splitDataView.TestSet);
                CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
                Console.WriteLine();
                Console.WriteLine("Model quality metrics evaluation");
                Console.WriteLine("--------------------------------");
                Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
                Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
                Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
                Console.WriteLine("=============== End of model evaluation ===============");
            }
            else
            {
                Console.WriteLine("Error! Sentiment Analyser not initalized.");
            }
        }

        private void UseModelWithSingleItem(BinaryClassData data)
        {
            if (initalized)
            {
                PredictionEngine<BinaryClassData, BinaryClassPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<BinaryClassData, BinaryClassPrediction>(model);


                var resultPrediction = predictionFunction.Predict(data);

                Console.WriteLine();
                Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

                Console.WriteLine();
                Console.WriteLine($"Sentiment: {resultPrediction.Value} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

                Console.WriteLine("=============== End of Predictions ===============");
                Console.WriteLine();
            }
            else
            {
                Console.WriteLine("Error! Sentiment Analyser not initalized.");
            }
        }

        public void UseModelWithBatchItems(IEnumerable<BinaryClassData> data)
        {
            if (initalized)
            {
                IDataView batchComments = mlContext.Data.LoadFromEnumerable(data);

                IDataView predictions = model.Transform(batchComments);

                // Use model to predict whether comment data is Positive (1) or Negative (0).
                IEnumerable<BinaryClassPrediction> predictedResults = mlContext.Data.CreateEnumerable<BinaryClassPrediction>(predictions, reuseRowObject: false);

                Console.WriteLine();

                Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

                foreach (var prediction in predictedResults)
                {
                    Console.WriteLine($"Sentiment: {prediction.Value} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
                }
                Console.WriteLine("=============== End of predictions ===============");
            }
            else
            {
                Console.WriteLine("Error! Sentiment Analyser not initalized.");
            }
        }
    }
}