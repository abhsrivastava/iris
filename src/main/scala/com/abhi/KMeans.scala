package com.abhi

import org.apache.spark._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg._

object KMeans extends App {
    Utils.deleteDirectory("model")
    val spark = Utils.getSparkSession()
    val iris = spark.sparkContext.textFile("src/main/resources/iris.csv")
    val vectors = iris.filter(_.nonEmpty).map{s => 
        Vectors.dense(s.split(',').take(4).map(_.toDouble))
    }
    val numberOfClusters = 3
    val numberOfIterations = 100
    val clusters = org.apache.spark.mllib.clustering.KMeans.train(
        vectors, numberOfClusters, numberOfIterations)
    val centers = clusters.clusterCenters
    val SSE = clusters.computeCost(vectors)
    println("++++++" + vectors.collect.map(clusters.predict))
    println(s"+++++ sum of Squared Errors $SSE")
    clusters.save(spark.sparkContext, "model/km-model")
    spark.close()
}