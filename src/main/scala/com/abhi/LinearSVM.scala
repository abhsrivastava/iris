package com.abhi

import org.apache.spark._
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation._

object LinearSVM extends App {
    Utils.deleteDirectory("model")
    val spark = Utils.getSparkSession()
    val (training, test) = Utils.loadData(spark.sparkContext)
    val numberOfIterations = 100
    val model = SVMWithSGD.train(training, numberOfIterations)
    model.clearThreshold()
    val scoreAndLabels = test.map{point => 
        val score = model.predict(point.features)
        (score, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    println(s"+++++++++ Area under ROC = ${auROC}")
    model
        .save(spark.sparkContext, "model/svm-model")

    spark.stop()
}