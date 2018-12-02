package com.abhi

import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.linalg._

object LinearRegression extends App {
    Utils.deleteDirectory("model")
    val spark = Utils.getSparkSession()
    val (training, test) = Utils.loadData(spark.sparkContext)
    val model = new LogisticRegressionWithLBFGS().setNumClasses(3).run(training)
    val predictionAndLabels = test.map{case LabeledPoint(label, features) => 
        val prediction = model.predict(features)
        (prediction, label)
    }
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println(s"++++++++ precision: ${precision}")  
    model.save(spark.sparkContext, "model/lr-model")
    spark.stop()
}