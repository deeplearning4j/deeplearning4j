package org.deeplearning4j.spark.ml.utils

/**
  * Created by derek.miller on 3/17/17.
  */
import org.apache.spark.sql.DataFrame

class DatasetFacade(val dataset: DataFrame) {
    def get : DataFrame = dataset
}

object DatasetFacade {
    def dataRows(dataset: DataFrame) : DatasetFacade = new DatasetFacade(dataset)
}

class DatasetBatchFacade(val dataset: java.util.List[DataFrame]){
    def get : java.util.List[DataFrame] = dataset
}

object DatasetBatchFacade {
    def dataRows(datasets: java.util.List[DataFrame]) = new DatasetBatchFacade(datasets)
}