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
