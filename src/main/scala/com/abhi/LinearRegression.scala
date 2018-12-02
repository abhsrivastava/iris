package com.abhi

import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.util._

object LinearRegression extends App {
    Utils.deleteDirectory("model")
    val spark = Utils.getSparkSession()
    val data = MLUtils.loadLibSVMFile(spark.sparkContext, "src/main/resources/iris-libsvm.txt")
    val splits = data.randomSplit(Array(.6, .4), seed=123L)
    val training = splits(0)
    val test = splits(1)
    val model = new LogisticRegressionWithLBFGS().setNumClasses(3).run(training)
    val predictionAndLabels = test.map{case LabeledPoint(label, features) => 
        val prediction = model.predict(features)
        (prediction, label)
    }
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    model.save(spark.sparkContext, "model/lr-model")
    spark.stop()
}