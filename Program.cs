
//
// Orginalkod av Mikael Hasselmalm
//Modifierad kod av Daniela Giovanella




// Install ML package with: dotnet add package Microsoft.ML
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

//
// Create the ML context
MLContext mlContext = new MLContext(seed: 1);

//Anger sökvägen till filen för dataset (i en textsräng)
string csvPath = "Titanic-Dataset.csv";
if (!File.Exists(csvPath))
{
    Console.WriteLine($"Couldn't find data file: {Path.GetFullPath(csvPath)}");
    return;
}

//
// Change color for the console text output
Console.ForegroundColor = ConsoleColor.Green;

//
// ML.NET read CSV data directly into IDataView.
// hasHeader: true because the file has a header row
// separatorChar: ',' are standard for CSV
// allowQuoting: true because titanic_small.csv have quotes around fields
// example "Braund, Mr. Owen Harris" there the name in itself contains a comma
// OBS! <PassengerData> is the class that describes the data schema for the CSV file and which features to read
IDataView data = mlContext.Data.LoadFromTextFile<PassengerData>(
    csvPath,
    hasHeader: true,
    separatorChar: ',',
    allowQuoting: true
);

List<PassengerData> passengers = mlContext.Data.CreateEnumerable<PassengerData>(data, reuseRowObject: false).ToList();
if (passengers.Count == 0)
{
    Console.WriteLine("The CSV file could not be parsed or contained no passengers.");
    return;
}

Console.WriteLine($"Read {passengers.Count} passengers from {csvPath}");
Console.WriteLine();

//
// Show the first 5 rows of the dataset
var preview = data.Preview(maxRows: 5);
Console.WriteLine("The first 5 rows of the dataset:");
foreach (var row in preview.RowView)
{
    foreach (var column in row.Values)
    {
        Console.Write($"{column.Key}: {column.Value}\t");
    }
    Console.WriteLine();
}
Console.WriteLine();

//
// Create the ML pipeline from titanic_small.csv:
//  "Sex" are text ("male"/"female") instead of 0/1, 
//  There are missing values in some rows. 
// 
// The pipeline creation transforms the raw data into a suitable format:
// - Sex are transformed with OneHotEncoding(numerical features)
// - ReplaceMissingValues are used for Age, Pclass, Parch and Fare to fill in missing values
var pipeline =
    mlContext.Transforms.Categorical.OneHotEncoding(
        outputColumnName: "SexEncoded",
        inputColumnName: nameof(PassengerData.Sex))
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "AgeFilled",
        inputColumnName: nameof(PassengerData.Age),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "ParentChildFilled",
        inputColumnName: nameof(PassengerData.Parch),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.DefaultValue))
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "PclassFilled",
        inputColumnName: nameof(PassengerData.Pclass),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "FareFilled",
        inputColumnName: nameof(PassengerData.Fare),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
    .Append(mlContext.Transforms.Concatenate(
        "Features",
        "AgeFilled",
        "SexEncoded",
        "ParentChildFilled",
        "PclassFilled",
        "FareFilled"
    ))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .Append(
        mlContext.BinaryClassification.Trainers
            .SdcaLogisticRegression(
                labelColumnName: nameof(PassengerData.Survived),
                featureColumnName: "Features"
            )
    );

//
// Validate with 5-fold cross validation.
var cvResults = mlContext.BinaryClassification.CrossValidate(
    data,
    pipeline,
    numberOfFolds: 5,
    labelColumnName: nameof(PassengerData.Survived)
);

double avgAccuracy = cvResults.Average(r => r.Metrics.Accuracy);
double avgAuc = cvResults.Average(r => r.Metrics.AreaUnderRocCurve);

Console.WriteLine($"Average Accuracy (5-fold CV): {avgAccuracy:P2}");
Console.WriteLine($"Average AUC (5-fold CV): {avgAuc:P2}");

//
// Create a train/test split (80/20) for a evaluation
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
// Count rows in each partition for informational purposes
long trainCount = mlContext.Data.CreateEnumerable<PassengerData>(split.TrainSet, reuseRowObject: false).LongCount();
long testCount = mlContext.Data.CreateEnumerable<PassengerData>(split.TestSet, reuseRowObject: false).LongCount();
Console.WriteLine();
Console.WriteLine($"Train/Test split: {trainCount} train rows, {testCount} test rows (80/20)");

//
// Train on the training partition and evaluate on the test partition
var modelOnTrain = pipeline.Fit(split.TrainSet);
var testPredictions = modelOnTrain.Transform(split.TestSet);
var testMetrics = mlContext.BinaryClassification.Evaluate(
    testPredictions,
    labelColumnName: nameof(PassengerData.Survived)
);
Console.WriteLine();
Console.WriteLine("Evaluation on 80/20 train-test split:");
Console.WriteLine($"Accuracy: {testMetrics.Accuracy:P2}");
Console.WriteLine($"AUC: {testMetrics.AreaUnderRocCurve:P2}");
Console.WriteLine($"F1 Score: {testMetrics.F1Score:P2}");
Console.WriteLine("Confusion matrix (on test set):");
Console.WriteLine(testMetrics.ConfusionMatrix.GetFormattedConfusionTable());

//
// Train the model on all data
// The cross validation are used to evaluate how well the model performs
var model = pipeline.Fit(data);

//
// Confusion matrix from the 5-fold CV on the full dataset
var finalPredictions = model.Transform(data);
var finalMetrics = mlContext.BinaryClassification.Evaluate(
    finalPredictions,
    labelColumnName: nameof(PassengerData.Survived)
);
Console.WriteLine();
Console.WriteLine("Confusion matrix, 5-fold CV(on full dataset):");
Console.WriteLine(finalMetrics.ConfusionMatrix.GetFormattedConfusionTable());

//ÅLDER
/* Create a new test passenger for prediction
PassengerData newPassenger = new()
{
    Age = 23f,
    Sex = "female",
    Pclass = 2f,
    Parch = 0f,
    Fare = 6f
};
*/
//Fråga om passagerarens ålder
Console.WriteLine("Ange ålder för passagerare:");

//Läser av användarens input och sparar det som en textsträng
string inputAge = Console.ReadLine();

//Kontrollerar om användarens input för ålder går att omvandla till deciamltal
//TryParse returnerar true om omvandling lyckas och false om den misslyckas
//Boolean variabel används för validering
//out gör att omvandling sparas i varibel age
bool validAge = float.TryParse(inputAge, out float age);

//KOntroll för om åldern inte kan omvandlas till ett tal
if (!validAge)
{
    //Visar felmeddelande om input inte är giltigt
    Console.WriteLine("Ogiltig ålder. Ange ett tal för ålder");
    //Avslutar programmet om input inte är giltig
    return;
}

//Kontroll för giltig ålder. Får inte va mindre än 0
if (age < 0)
{
    //Visar felmeddelande om tal för åldern är negativ
    Console.WriteLine("Åldern får inte vara ett negativt tal");
    //Avslutar programmet vid fel input
    return;
}

//NYtt objket för ny passagerare
//Skapar ett objekt av klassen PassengerData för ny passagerare
PassengerData newPassenger = new()
{
    //Ålder (klar)
    Age = age,
    //Ej klara
    Sex = "female",
    Pclass = 2f,
    Parch = 0f,
    Fare = 6f

};


// Create a prediction engine
var predictionEngine =
    mlContext.Model.CreatePredictionEngine<PassengerData, PassengerPrediction>(model);

//
// Predict survival for the created test passenger
PassengerPrediction prediction = predictionEngine.Predict(newPassenger);

Console.WriteLine();
Console.WriteLine("Ny passagerare:");
Console.WriteLine($"Ålder: {newPassenger.Age}");
Console.WriteLine($"Kön: {newPassenger.Sex}");
Console.WriteLine($"Klass: {newPassenger.Pclass}");
Console.WriteLine($"Antal föräldrar/barn: {newPassenger.Parch}");
Console.WriteLine($"Biljettpris: {newPassenger.Fare}");

Console.WriteLine();
Console.WriteLine(
    prediction.Survived
        ? "Modellen tror att passageraren överlever."
        : "Modellen tror att passageraren inte överlever."
);

Console.WriteLine($"Sannolikhet: {prediction.Probability:P1}");

//
// Reset text color to default
Console.ResetColor();

//
// CSV column order in the dataset (titanic_small.csv):
// PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
public class PassengerData
{
    [LoadColumn(1)] public bool Survived { get; set; }
    [LoadColumn(2)] public float Pclass { get; set; }
    [LoadColumn(4)] public string Sex { get; set; } = "";
    [LoadColumn(5)] public float Age { get; set; }
    [LoadColumn(7)] public float Parch { get; set; }
    [LoadColumn(9)] public float Fare { get; set; }
}

// Class description of the prediction output
public class PassengerPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Survived { get; set; }

    public float Probability { get; set; }

    public float Score { get; set; }
}
