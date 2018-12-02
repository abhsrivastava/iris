package com.abhi

import java.io._
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.util._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

object Utils {
    def deleteFile(f: File) : Unit = {
        if (f.isDirectory) {
            for(fc <- f.listFiles) {
                deleteFile(fc)
            }
            f.delete()
        } else {
            f.delete()
        }
    }
    def deleteDirectory(d: String) : Unit = deleteFile(new File(s"./$d"))
    def getSparkSession() : SparkSession = {
        SparkSession
            .builder
            .master("local[*]")
            .appName("Linear Support Vector Machine")
            .getOrCreate()        
    }
    def loadData(sc: SparkContext) : (RDD[LabeledPoint], RDD[LabeledPoint]) = {
        val data = MLUtils.loadLibSVMFile(sc, "src/main/resources/iris-libsvm.txt")
        val splits = data.randomSplit(Array(.7, .3), seed=123L)
        (splits(0), splits(1))
    }
}