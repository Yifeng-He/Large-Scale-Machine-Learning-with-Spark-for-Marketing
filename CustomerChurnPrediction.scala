import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object CustomerChurnPrediction {
  
  // function to set the log level
  def setupLogging() = {
    import org.apache.log4j.{Level, Logger}   
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)   
  }
  
  // the format of each row in the DataFrame. An example of a line is:
  // KS, 128, 415, 382-4657, no, yes, 25, 265.1, 110, 45.07, 197.4, 99, 16.78, 244.7, 91, 11.01, 10, 3, 2.7, 1, False.
  // The columns are:
  // state, account_length, area_code, phone_number, international_plan, voice_mail_plan, number_vmail_messages
  // total_day_minutes, total_day_calls, total_day_charge, total_eve_minutes, total_eve_calls, total_eve_charge, 
  // total_night_minutes, total_night_calls, total_night_charge, total_intl_minutes, total_intl_calls, total_intl_charge, 
  //number_customer_service_calls, churned
  case class Record(label : String, account_length: Float, international_plan: String, number_vmail_messages: Int,
      total_day_calls : Int, total_day_charge : Float,
      total_eve_calls : Int,  total_eve_charge : Float,
      total_night_calls: Int, total_night_charge : Float,
      total_intl_calls: Int, total_intl_charge : Float, number_customer_service_calls: Int )
  
  // the function to convert each line of string into a structured row
  def convertLinetoRow(line: String): Record = {
    val fields = line.split(", ")
    assert(fields.size == 21)
    Record(fields(20).toString, fields(1).toFloat, fields(4).toString, fields(6).toInt, fields(8).toInt, fields(9).toFloat,
        fields(11).toInt, fields(12).toFloat, fields(14).toInt, fields(15).toFloat, fields(17).toInt, fields(18).toFloat, fields(19).toInt)
  }
 
   // the main function
   def main(args: Array[String]): Unit = {
    
     // step 1: set up spark session 
     val spark =SparkSession.builder.config(key="spark.sql.warehouse.dir", value="file:///C:/Temp").master("local")
     .appName("CustomerChurnPrediction").getOrCreate() 
     // set up log level
    setupLogging()
    
    // step 2: load the data into dataFrame
    val dataset_lines = spark.read.textFile("C:/data/customer_churn_data.txt")
    // convert the DataSet of lines to DataFrame of Rows
    import spark.implicits._
    val data_raw = dataset_lines.map(convertLinetoRow).toDF
    //data.printSchema()
    data_raw.show(1)
    
    // step 3: data pre-processing
    // encode categorical labels to index number
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel")
    val indexed_data1 = labelIndexer.fit(data_raw).transform(data_raw)
    val internationalPlanIndexer = new StringIndexer().setInputCol("international_plan").setOutputCol("indexed_international_plan")
    val indexed_data2 = internationalPlanIndexer.fit(indexed_data1).transform(indexed_data1)
    //indexed_data2.show(1)
    
    // assemble the features vector
    val assembler = new VectorAssembler().setInputCols(Array("account_length", "indexed_international_plan", "number_vmail_messages", 
        "total_day_calls", "total_day_charge", "total_eve_calls", "total_eve_charge", "total_night_calls", "total_night_charge",
        "total_intl_calls", "total_intl_charge", "number_customer_service_calls"))
        .setOutputCol("features")
    val data_features = assembler.transform(indexed_data2)
    //data_features.show(1)
    
    // standardize the features to standard normal distribution (mean=0, variance=1)
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(true)
    val data_scaled_features = scaler.fit(data_features).transform(data_features)
    // final data with labels and features after pro-processing
    val data = data_scaled_features.select("indexedLabel", "scaledFeatures").toDF("label", "features")
    println (s"The number of data samples is ${data.count()}.")
    data.show(1)
    
    // step 4: split the data into training and test sets
    val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))
    
    // step 5: classifier model
    val random_forest = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features")
    // check the parameters descriptions 
    //println(random_forest.explainParams())
    
    // step 6: model training and parameter tuning
    // parameter set
    val paramGrid = new ParamGridBuilder()
        .addGrid(random_forest.maxDepth, Array(3, 5, 10))
        .addGrid(random_forest.numTrees, Array(10, 20, 50)).build()
    
    // CrossValidator requires: 1) an estimator, 2) a set of ParamMaps, and 3) an evaluator.
    val cross_validator = new CrossValidator()
      .setEstimator(random_forest)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)  // Use 3+ in practice
    // training
    val cvModel = cross_validator.fit(trainingData)
    
    // step 7: prediction and evaluation
    val predictions = cvModel.transform(testData)
    predictions.printSchema()
    predictions.show(1)
    // evaluate
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy = " + accuracy) //Accuracy = 0.9567779960707269

    // step 8: check the learned model
    val map_parameters = cvModel.bestModel.explainParams()
    println(map_parameters)   
    val learned_model = cvModel.bestModel.asInstanceOf[RandomForestClassificationModel]
    println("Learned classification random forest model:\n" + learned_model.toDebugString)
     
    // save the model that has been trained
    //cvModel.write.overwrite().save("C:/data/random_forest_model")    
         
    spark.stop()
   }  
}
