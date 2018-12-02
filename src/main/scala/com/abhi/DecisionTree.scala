package com.abhi

import org.apache.spark._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.mllib.tree.configuration._
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.evaluation._

object DecisionTree extends App {
    Utils.deleteDirectory("model")
    val spark = Utils.getSparkSession()
    val (training, test) = Utils.loadData(spark.sparkContext)
    val strategy = new Strategy(Classification, Gini, 10, 3, 10)
    val decisionTree = new org.apache.spark.mllib.tree.DecisionTree(strategy)
    val model = decisionTree.run(training)
    val predictionAndLabels = test.map{case LabeledPoint(label, features) => 
        val prediction = model.predict(features)
        (label, prediction)
    }
    model.save(spark.sparkContext, "model/dt-model")
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println(s"++++++++ precision: ${precision}")    
    spark.stop()
}