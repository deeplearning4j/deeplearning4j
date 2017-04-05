package org.deeplearning4j.spark.ml.utils


import org.apache.spark.sql.Dataset

class DatasetFacade(val dataset: Dataset[_]){
    def get : Dataset[_] = dataset
}

object DatasetFacade {
    def dataRows(dataset: Dataset[_]) : DatasetFacade = new DatasetFacade(dataset)
}

class DatasetBatchFacade(val dataset: java.util.List[Dataset[_]]){
    def get : java.util.List[Dataset[_]] = dataset
}

object DatasetBatchFacade {
    def dataRows(datasets: java.util.List[Dataset[_]]) = new DatasetBatchFacade(datasets)
}
