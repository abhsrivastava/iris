package com.abhi

import org.apache.spark._
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.util._

object LinearSVM extends App {
    Utils.deleteDirectory("model")
    val spark = Utils.getSparkSession()
    val data = MLUtils
                    .loadLibSVMFile(
                        spark.sparkContext, 
                        "src/main/resources/iris-libsvm.txt")
    val splits = data.randomSplit(Array(.6, .4), seed=123L)
    val training = splits(0)
    val test = splits(1)

    val numberOfIterations = 100
    val model = SVMWithSGD.train(training, numberOfIterations)
    model.clearThreshold()
    val scoreAndLabels = test.map{point => 
        val score = model.predict(point.features)
        (score, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    println(s"Area under ROC = ${auROC}")
    model
        .save(spark.sparkContext, "model/svm-model")

    spark.stop()
}