package com.abhi

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object RandomForest extends App {
    val spark = Utils.getSparkSession()
    import spark.implicits._

    val lines = spark
                    .sparkContext
                    .textFile("src/main/resources/iris.csv")
                    .filter(_.nonEmpty)
                    .flatMap(s => s.split("\n"))
                    .map(_.split(","))
                    .mapPartitionsWithIndex{case (idx, iter) => if (idx == 0) iter.drop(1) else iter}
                    .map{row => (Vectors.dense(row(0).toDouble, row(1).toDouble, row(2).toDouble, row(3).toDouble), row(4))}   

    val df = spark
                .createDataFrame(lines)
                .toDF("iris-features-column","iris-species-column")
    df.show()
    // let us build the pipeline
    val indexer = new StringIndexer()
                    .setInputCol("iris-species-column")
                    .setOutputCol("label")
    val splitDataSet = df.randomSplit(Array(.7, .3), seed = 98765L)
    val (train, test) = (splitDataSet(0), splitDataSet(1))
    val randomForestClassifier = new RandomForestClassifier()
                                    .setFeaturesCol("iris-features-column")
                                    .setFeatureSubsetStrategy("sqrt")
    val rfNum_Trees = randomForestClassifier.setNumTrees(15)
    println("Hyper Parameter num_trees is: " + rfNum_Trees.numTrees)
    //confirm that the classifier has a default value set
    println("Is Max-Depth for classifier set? - " + rfNum_Trees.hasDefault(rfNum_Trees.numTrees))
    println("Default Max_Depth set on classifier is - " + rfNum_Trees.getOrDefault(rfNum_Trees.numTrees))

    val pipeline = new Pipeline()
                        .setStages(Array[PipelineStage](indexer) ++ Array[PipelineStage](randomForestClassifier))

    val gridBuilder = new ParamGridBuilder()
                        .addGrid(rfNum_Trees.numTrees, Array(8,16,24,32,40,48,56,64,72,80,88,96))

    //set this default parameter in the classifier's embedded param map
    val rfMax_Depth = rfNum_Trees.setMaxDepth(2)
    println("Hyper Parameter max_depth is: " + rfMax_Depth.maxDepth)

    //confirm that the classifier has a default value set
    println("Is max_depth for classifier set? - " + rfMax_Depth.hasDefault(rfMax_Depth.maxDepth))
    println("Default max_depth set on classifier is - " + rfMax_Depth.getOrDefault(rfMax_Depth.maxDepth))

    //Now lets add our MAX_DEPTH hyper parameter to the param grid
    val gridBuilder2 = gridBuilder.addGrid(rfMax_Depth.maxDepth, Array(4,10,16,22,28))

    val rfImpurity = rfMax_Depth.setImpurity("gini")
    println("Hyper Parameter Impurity value is: " + rfImpurity.impurity)

    //confirm that the classifier has a default value set
    println("Is Impurity for classifier set?  - " + rfImpurity.hasDefault(rfImpurity.impurity))
    println("Default Impurity set on classifier is - " + rfImpurity.getOrDefault(rfImpurity.impurity))

    //Now lets add our IMPURITY hyper parameter to the param grid
    val gridBuilder3 = gridBuilder2.addGrid(rfImpurity.impurity, Array("gini", "entropy"))

    println("Confirming that Default Max_Depth set on classifier is - " + rfImpurity.getOrDefault(rfNum_Trees.numTrees))
    println("Confirming that Default max_depth set on classifier is - " + rfImpurity.getOrDefault(rfMax_Depth.maxDepth))
    println("Confirming that Default Impurity set on classifier is - " + rfImpurity.getOrDefault(rfImpurity.impurity))

    val finalParamGrid: Array[ParamMap] = gridBuilder3.build()

    val validatedTestResults = new TrainValidationSplit()
        .setSeed(1234567L)
        .setEstimatorParamMaps(finalParamGrid)
        .setEstimator(pipeline)
        .setEvaluator(new MulticlassClassificationEvaluator())
        .setTrainRatio(.85)
        .fit(train)
        .transform(test)

    val validatedTestResultsDataset = validatedTestResults.select("prediction", "label")
    println("Validated TestSet Results Dataset is:  " + validatedTestResultsDataset.take(10))

    val validatedRDD2 = validatedTestResultsDataset.rdd.collect {
            case Row(predictionValue: Double, labelValue: Double) => (predictionValue,labelValue)
    }

    val modelOutputAccuracy = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setMetricName("accuracy")
        .setPredictionCol("prediction").evaluate(validatedTestResultsDataset)

    println("Accuracy of Model Output results on the test dataset: " + modelOutputAccuracy)    

    val multiClassMetrics = new MulticlassMetrics(validatedRDD2)
    val accuracyMetrics = (multiClassMetrics.accuracy, multiClassMetrics.weightedPrecision)
    val accuracy = accuracyMetrics._1
    val weightedPrecsion = accuracyMetrics._2
    println("Accuracy (precision) is " + accuracy + " Weighted Precision is: " + weightedPrecsion)
    spark.close()
}