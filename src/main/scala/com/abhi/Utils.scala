package com.abhi

import java.io._
import org.apache.spark.sql.SparkSession

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
}