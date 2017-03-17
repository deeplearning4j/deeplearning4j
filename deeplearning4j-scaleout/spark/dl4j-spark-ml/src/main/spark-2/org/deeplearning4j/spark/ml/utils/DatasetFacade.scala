package org.deeplearning4j.spark.ml.utils


import org.apache.spark.sql.Dataset

class DatasetFacade(val dataset: Dataset[_]){
    def get : Dataset[_] = dataset
}

object DatasetFacade {
    def dataRows(dataset: Dataset[_]) : DatasetFacade = new DatasetFacade(dataset)
}
