
//
//Orginalkod av Mikael Hasselmalm
//Modifierad kod av Daniela Giovanella




//Imorterar funktioner från ML.net
//Används för att skapa ML-miljö, hantera data och features
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;


//Skapar ett MLContext- objekt som används för ML
//Seed 1 används för att göra slumpmässiga moment mer återskapningsbar
MLContext mlContext = new MLContext(seed: 1);

//Sparar sökvägen till filen för dataset (i en textsräng)
string csvPath = "Titanic-Dataset.csv";

//KOntroll för om filen inte finns
if (!File.Exists(csvPath))
{
    //Om filen inte hittas skrivs felmeddelande ut
    //Path.GetFullPath = Hela sökvägen
    Console.WriteLine($"Couldn't find data file: {Path.GetFullPath(csvPath)}");

    //Avslutar programmet om fil inte hittas
    return;
}



//Läser in csv-filen och sparar dess data i ett IDataView-objekt som är ett format som används för att hantera data i ML.Net
//PassengerData beskriver vilka kolumner som skaläsas in och vad de har för datatyper
IDataView data = mlContext.Data.LoadFromTextFile<PassengerData>(
    csvPath,
    //Csv-filen börjar med en rubrikrad
    hasHeader: true,
    //Kolumnerna skiljs åt med kommatecken
    separatorChar: ',',
    //Tillåter citat som kan innehålla kommatecken
    allowQuoting: true
);

//Gör datan till en lista med PassengerData- objekt där varje objekt är en passagerare
List<PassengerData> passengers = mlContext.Data.CreateEnumerable<PassengerData>(data, reuseRowObject: false).ToList();

//KOntrollerar om listan innehåller några passagerare. Vid 0:
if (passengers.Count == 0)
{
    //Skriver ut felmeddelande om inga passagerare kunde läsas in
    Console.WriteLine("The CSV file could not be parsed or contained no passengers.");
    //Avslutar programmet om inga passagerare hittats
    return;
}

//Om passagerare finns skrivs antal passagerare och vilken fil som använts ut
Console.WriteLine($"Read {passengers.Count} passengers from {csvPath}");
Console.WriteLine();

//Visar de första fem raderna i datasetet (förhandsvisning)
var preview = data.Preview(maxRows: 5);
Console.WriteLine("The first 5 rows of the dataset:");

//Loopar igenom varje kolumn i förhandsvisningen
foreach (var row in preview.RowView)
{
    //Loopar igenom varje kolumn i aktuell rad
    foreach (var column in row.Values)
    {
        //Skriver ut kolumnens namn och värde
        //\t används för tab. Formatering av kolumner
        Console.Write($"{column.Key}: {column.Value}\t");
    }
    Console.WriteLine();
}
Console.WriteLine();

//PIPELINE
//Skapar pipeline  från csv- fil som innehåller stegen för attförbetedea dataset och träna ML-modellen:
//Sparas i varibeln pipeline
//Gör om text till numeriska värden, hanterar saknade värden och samlar features i en kolumn
//  

//Omvandlar textvärde som Sex (kön) till numeriskt värde med OneHotEncoding
//Ny kolumn med omvandlad värde får då namet SexEncode
var pipeline =
    mlContext.Transforms.Categorical.OneHotEncoding(
        outputColumnName: "SexEncoded",
        inputColumnName: nameof(PassengerData.Sex))

    //Hanterar saknade åldersvärden genom att ersätta dessa med medlevärdet för ålder
    //Bearbetad kolumn får namnet AgedFilled
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "AgeFilled",
        inputColumnName: nameof(PassengerData.Age),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))

    //hanterar saknade värden i Parch. Antal föräldrar eller barn som passagerare har med sig
    //DefaultValue ersätter det saknade värdet med 0 
    //Ny kolumn får namnet ParentChildFilled
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "ParentChildFilled",
        inputColumnName: nameof(PassengerData.Parch),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.DefaultValue))

    //Hanterar saknade värden i Pclass
    //Ersätter saknat värde med medelvärde i biljettklass-kolumnen
    //Ny kolumn får namnet PclassFilled
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "PclassFilled",
        inputColumnName: nameof(PassengerData.Pclass),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))

    //Hanterar saknat värde i Fare
    //Ersätter saknat värde med medelvärde i biljettpris-kolumnen
    //Ny kolumn får namnet FareFilled
    .Append(mlContext.Transforms.ReplaceMissingValues(
        outputColumnName: "FareFilled",
        inputColumnName: nameof(PassengerData.Fare),
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))

    //Samlar alla feauters i en gemensam kolumn som heter Features
    //Information som modell lär sig samband och mönster för att överleva eller inte
    .Append(mlContext.Transforms.Concatenate(
        "Features",
        "AgeFilled",
        "SexEncoded",
        "ParentChildFilled",
        "PclassFilled",
        "FareFilled"
    ))

    //Normaliserar värdena i Features till ett jämförbart numeriskt intervall som hjälper modellen hantera features med olika skalor
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))

    //Lägger till algoritm som ska träna midellen
    //SdLogisticRegression används för binär klassificering
    //Detta för att modell ska förutsäga ett av två resultat, överlevde eller inte
    .Append(
        mlContext.BinaryClassification.Trainers
            .SdcaLogisticRegression(
                labelColumnName: nameof(PassengerData.Survived),
                featureColumnName: "Features"
            )
    );

//Utvärderar pipeline med 5-fold cross validation
//Sparar desultat i CvResults
/*
Dataset delas upp i fem delar och tränas fem gånger där varje del används som testdata i
en av omgångarna. Detta för att se hur modellen fungerar på data som den inte tränats på i varje omgång
*/
var cvResults = mlContext.BinaryClassification.CrossValidate(
    data,
    pipeline,
    numberOfFolds: 5,
    labelColumnName: nameof(PassengerData.Survived)
);

//Beräknar genomsnittlig accuracy från dom fem omgångarna
//Accuracy är delen förutsägelser som blev rätt
double avgAccuracy = cvResults.Average(r => r.Metrics.Accuracy);
//Beräknar det genomsnittliga auc-värdet från de fem omgångarna
//Auc mäter hur bra modellen kan skilja mellan de två klasserna
double avgAuc = cvResults.Average(r => r.Metrics.AreaUnderRocCurve);


//Skriver ut dom genomsnittliga utvärderingsresultaten
//P2 formaterar värdena som procent med två decimaler
Console.WriteLine($"Average Accuracy (5-fold CV): {avgAccuracy:P2}");
Console.WriteLine($"Average AUC (5-fold CV): {avgAuc:P2}");

//DELA UPP DATA
//Delar upp datan tränigsdataset och testdata
//Träningsdata står för 80% och 20% för utvärdering
//Seed 1 gör den slumpmässiga uppdelningen mer återanvändningsbar
//spars i variabel split
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);

//Räknas antalet rader i träningsdelen
//CreateEnumerable gör det möjligt att gå igenom raderna som PassengerData-objekt
//LongCount räknar antalet rader
long trainCount = mlContext.Data.CreateEnumerable<PassengerData>(split.TrainSet, reuseRowObject: false).LongCount();

//Räknar antalet rader i testdelen på samma sätt
long testCount = mlContext.Data.CreateEnumerable<PassengerData>(split.TestSet, reuseRowObject: false).LongCount();

//Skriver ut antalet rader från varje del
Console.WriteLine();
Console.WriteLine($"Train/Test split: {trainCount} train rows, {testCount} test rows (80/20)");

//TRÄNA MODELL OCH UTVÄRDERA DATA

//Tränar pipelinen på träningsdelen
//Fit bearbetar träningsdatan och lär modellen samband mellan Features och Survived
var modelOnTrain = pipeline.Fit(split.TrainSet);

//Använder den tränade modellen på testdatan
//Transform skapar förutsägelser för passagerare som modellen inte tränades på
var testPredictions = modelOnTrain.Transform(split.TestSet);

//Jämför modellens förutsägelser med värdena för Survived och beräknar olika utvärderingsmått
var testMetrics = mlContext.BinaryClassification.Evaluate(
    testPredictions,
    labelColumnName: nameof(PassengerData.Survived)
);


Console.WriteLine();
//Skriver ut resultaten från utvärderingen på testdatan
Console.WriteLine("Evaluation on 80/20 train-test split:");
//Accuracy visar andelen förutsägelser som blev rätt
Console.WriteLine($"Accuracy: {testMetrics.Accuracy:P2}");
//AUC visar hur bra modellen skiljer mellan de två klasserna
Console.WriteLine($"AUC: {testMetrics.AreaUnderRocCurve:P2}");
//F1 Score kombinerar precision och recall till ett mått
Console.WriteLine($"F1 Score: {testMetrics.F1Score:P2}");
//Skriver ut en confusion matrix som visar hur förutsägelserna fördelas mellan korrekt och felaktig klassificering
Console.WriteLine("Confusion matrix (on test set):");
Console.WriteLine(testMetrics.ConfusionMatrix.GetFormattedConfusionTable());

//TRÄNA MODELL PÅ HELA DATASETET
//Modell som används för att förutsäga överlevnad för en ny passagerare som användare matear in
var model = pipeline.Fit(data);

//Använder modellen på samma dataset som den tränades på
//Resultatet visar hur modellen klassificerar träningsdatan,men är inte en oberoende utvärdering av nya passagerare
var finalPredictions = model.Transform(data);


//Jämför förutsägelserna med värdena i Survived
var finalMetrics = mlContext.BinaryClassification.Evaluate(
    finalPredictions,
    labelColumnName: nameof(PassengerData.Survived)
);

//Skriver ut confusion matrix för träningsdatan
Console.WriteLine();
Console.WriteLine("Confusion matrix, 5-fold CV(on full dataset):");
Console.WriteLine(finalMetrics.ConfusionMatrix.GetFormattedConfusionTable());



///// NY PASSAGERARE - Baserad på användarinput ///////

//ÅLDER

//Variabel som har koll på om åldern är giltig
bool validAge = false;

//Variabel som ska innehålla passagerarens ålder
float age = 0;

//Loop som upprepar frågan om ålder så länge åldern inte är giltig
while (!validAge)

{
    //Fråga om passagerarens ålder
    Console.WriteLine("Ange ålder för passagerare:");

    //Läser av användarens input och sparar det som en textsträng
    //?? "" används för att hantera nullvärde med en tom textsträng
    string inputAge = Console.ReadLine() ?? "";


    //Kontrollerar om användarens input för ålder går att omvandla till deciamltal
    //TryParse returnerar true om omvandling lyckas och false om den misslyckas
    //out gör att omvandling sparas i varibel age
    validAge = float.TryParse(inputAge, out age);

    //KOntroll för om åldern inte kan omvandlas till ett tal
    if (!validAge)
    {
        //Visar felmeddelande om input inte är giltigt
        Console.WriteLine("Ogiltig ålder. Ange ett tal för ålder");
    }

    //Kontroll för giltig ålder. Får inte va mindre än 0
    else if (age < 0)
    {
        //Visar felmeddelande om tal för åldern är negativ
        Console.WriteLine("Åldern får inte vara ett negativt tal");

        //Frågan ska upprepas så länge validAge är false
        validAge = false;

    }
}

//KÖN

//Variabel som håller koll på om könet är giltigt
bool validSex = false;

//Variabel som ska innehålla passagerarens kön
string inputSex = "";

//Loop som upprepar frågan om kön så länge kön inte är giltig
while (!validSex)
{

    //Frågar användare om passagerares kön
    Console.WriteLine("Ange kön. male eller female:");

    //LÄser av användarens input och sparar den som en textsträng
    //?? "" används för att hantera nullvärde med en tom textsträng
    inputSex = Console.ReadLine() ?? "";

    //Kontrollerar ifall användaren inet angett nåt av de två alternativen
    //male eller female
    if (inputSex != "male" && inputSex != "female")
    {
        //Visar felmeddelande om input inte stämmer
        Console.WriteLine("Oglitigt kön. Du måste ange male eller female");
    }
    else
    {
        //Är kön giltigt kan loop avslutas
        validSex = true;
    }
}

//KLASS

//Variabel som håller koll på om klass är giltigt
bool validClass = false;

//Variabel som ska innehålla passagerarens klass
float passengerClass = 0;

//Loop som upprepar frågan om klass så länge klass inte är giltig
while (!validClass)
{

    //Frågar användaren efter passagerarens klass
    Console.WriteLine("Ange passagerarklass. 1, 2 eller 3:");

    //LÄser av användarens input och sparar den som en textsträng
    string inputClass = Console.ReadLine() ?? "";


    //Kontrollerar om användarens input för klass går att omvandla till deciamltal
    //TryParse returnerar true om omvandling lyckas och false om den misslyckas
    //out gör att omvandling sparas i varibel passengerClass
    validClass = float.TryParse(inputClass, out passengerClass);

    //KOntroll för om åldern inte kan omvandlas till ett tal
    if (!validClass)
    {
        //Visar felmeddelande om input inte är giltigt
        Console.WriteLine("Ogiltig klass. Ange 1, 2 eller 3");

    }

    //KOntroll för om användarinput för klass inte är 1, 2 eller 3
    else if (passengerClass != 1 && passengerClass != 2 && passengerClass != 3)
    {
        //Visar felmeddelande vid fel input
        Console.WriteLine("Ogiltig klass. Ange 1, 2 eller 3");

        //Upprepar frågan så länge validclass är false
        validClass = false;

    }
}

//FÖRÄLDRAR OCH BARN

//Parch = HUr många barn och föärldra som reste tillsammans

//Variabel som håller koll på om antalet är giltigt
bool validParch = false;

//Variabel som ska innehålla antalet föräldrar och barn
int parch = 0;

//Loop som upprepar frågan så länge antalet inte är giltigt
while (!validParch)
{

    //Fråga användaren hur många barn eller förädrar som reste med passagerare
    Console.WriteLine("Hur många föräldrar och barn reste tillsammans med passageraren? (Ange 0 om passageraren reste utan föräldrar eller barn)");

    //LÄser av användarens input och sparar som textsträng
    //?? "" används för att hantera nullvärde med en tom textsträng
    string inputParch = Console.ReadLine() ?? "";

    //Kontrollerar om användarens input för parch går att omvandla till deciamltal
    //TryParse returnerar true om omvandling lyckas och false om den misslyckas
    //int används för att vi vill få ut ett heltal
    //out gör att omvandling sparas i varibel parch
    validParch = int.TryParse(inputParch, out parch);

    //KOntroll för om åldern inte kan omvandlas till ett tal
    if (!validParch)
    {
        //Visar felmeddelande om input inte är giltigt
        Console.WriteLine("Ogiltig antal. Ange ett heltal som ex: 0, 1 eller 2");
    }

    //KOntroll för om antalet är ett nagativt tal
    else if (parch < 0)
    {
        //Visar felmeddelande om antalet är negativt
        Console.WriteLine("Antalet måste vara 0 eller större");

        //Frågan upprepas så länge validParch är false
        validParch = false;
    }
}

//BILJETTPRIS

//Variabel som håller koll på om biljettpriset är giltigt
bool validFare = false;

//Variabel som ska innehålla biljettpriset
float fare = 0;

//Loop som upprepar frågan så länge biljettpriset inte är giltigt
while (!validFare)
{

    //Fråga om biljettpris
    Console.WriteLine("Ange biljettpris:");

    //Läser av användarens input och sparar det som en textsträng
    //?? "" används för att hantera nullvärde med en tom textsträng
    string inputFare = Console.ReadLine() ?? "";

    //Kontrollerar om användarens input för biljettpris går att omvandla till deciamltal
    //TryParse returnerar true om omvandling lyckas och false om den misslyckas
    //Boolean variabel används för validering
    //out gör att omvandling sparas i varibel fare
    validFare = float.TryParse(inputFare, out fare);

    //KOntroll för om biljettpris inte kan omvandlas till ett tal
    if (!validFare)
    {
        //Visar felmeddelande om input inte är giltigt
        Console.WriteLine("Ogiltigt biljettpris. Ange ett tal:");
    }

    //Kontroll för giltigt pris. Får inte va mindre än 0
    else if (fare < 0)
    {
        //Visar felmeddelande om tal för biljettpris är negativ
        Console.WriteLine("Biljettpris får inte vara ett negativt tal");
        //Upprepar frågan så länge validFare är false
        validFare = false;

    }
}


//NYtt objket för ny passagerare
//Skapar ett objekt av klassen PassengerData med features för ny passagerare
PassengerData newPassenger = new()
{
    //Ålder
    Age = age,
    //Kön
    Sex = inputSex,
    //Klass
    Pclass = passengerClass,
    //Antal föräldrar och barn
    Parch = parch,
    //Biljettpris
    Fare = fare

};


//Skapar variabeln predictionEngine som användes för att göra förutsägelser med tränad MLmodell
//mlContext.Model används för att komma åt modellen
//createPredictionEngine anger vilken data som ska användas
//Datan som används är från klassen PassengerData
//PassengerPrediction är klassen som tar emot resultatet från förutsägelsen
//model är vilken modell som ska användas
var predictionEngine =
    mlContext.Model.CreatePredictionEngine<PassengerData, PassengerPrediction>(model);

//Använder den tränade modellen för att göra förutsägelser för ny passagerare
//newPassenger innehåller de uppgifter som användaren angett
//Predict skeickar uppgifterna till modellen för att göra förutsägelsen
//Resultat sparas i variabeln prediction
PassengerPrediction prediction = predictionEngine.Predict(newPassenger);

//Skriver ut tom rad
Console.WriteLine();
//Skriver ut rubrik för ny passagerare
Console.WriteLine("Ny passagerare:");
//Skriver ut ålder från objektet newPassenger
Console.WriteLine($"Ålder: {newPassenger.Age}");
//Skriver ut kön...
Console.WriteLine($"Kön: {newPassenger.Sex}");
//Skriver ut klass...
Console.WriteLine($"Klass: {newPassenger.Pclass}");
//Skriver ut antal föräldrear o barn...
Console.WriteLine($"Antal föräldrar/barn: {newPassenger.Parch}");
//Skriver ut biljettpris
Console.WriteLine($"Biljettpris: {newPassenger.Fare}");

//Skriver ut tom rad
Console.WriteLine();

//KOntrollerar värdet från prediction:Survived för att välja vad som ska skrivas ut i konsol
//Om suvived är true eller false
Console.WriteLine(
    prediction.Survived
        ? "Modellen tror att passageraren överlever."
        : "Modellen tror att passageraren inte överlever."
);


//Visar modellens beräknande sannolikhet för den förutsagda klassen
//Probability innehåller värder mellan 0 och 1
//P1 formaterar värdet som procent med en decimal
Console.WriteLine($"Sannolikhet: {prediction.Probability:P1}");



//Skapar klassen PassengerData
//Beskriver vilka uppgifter som finns för en passagerare
//Public då den används i andra delar av programmet
//LoadColumn anger vilken kolumn i csv-filen som ska kopplas till respektive egenskap
//get - läser värdet, set - tilldelar eller ändrar värdet
public class PassengerData
{
    [LoadColumn(1)] public bool Survived { get; set; }
    [LoadColumn(2)] public float Pclass { get; set; }
    [LoadColumn(4)] public string Sex { get; set; } = ""; // "" ger en tom textsträng som startvärde
    [LoadColumn(5)] public float Age { get; set; }
    [LoadColumn(7)] public float Parch { get; set; }
    [LoadColumn(9)] public float Fare { get; set; }
}

//Klass som beskriver resultatet som kommer tillbaka från ML modellen
//Publik då den används på fler ställen i programmet
public class PassengerPrediction
{
    //Kopplar egenskapen Survived till kolumnen Predictlabel i modellens resultat
    //PredictLabel innehåller förutsägelsen om passageraren överlever eller inte
    //bool då resultatet antingen är true eller false
    [ColumnName("PredictedLabel")]
    public bool Survived { get; set; }

    //Egenskap med modellens beräknade sannorlikhet
    public float Probability { get; set; }

    //Egenskap med mdellens beräknade poäng för förutsägelsen
    //Används för att komma fram till förutsägelsen
    public float Score { get; set; }
}
