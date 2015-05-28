import org.apache.spark.sql.sources._

//sqlContext.sql("CREATE TEMPORARY TABLE oneToTenPruned USING org.apache.spark.sql.sources.PrunedScanSource OPTIONS (from '1', to '10')")


val df = sqlContext.baseRelationToDataFrame(SimplePrunedScan(1, 10)(sqlContext))
val df2 = df.select('a,'c)

import org.apache.spark.ml.feature._
val tokenizer = new Tokenizer()
tokenizer.setInputCol("c")
tokenizer.setOutputCol("d")
val df3 = tokenizer.transform(df)
