/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.spark.ml.impl


import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.ml.utils.{DatasetFacade, SchemaUtils}
import org.nd4j.linalg.factory.Nd4j


class AutoEncoder(uid: String) extends AutoEncoderWrapper[AutoEncoder, AutoEncoderModel](uid){

    def this() {
        this(Identifiable.randomUID("dl4j"))
    }

    override def mapVectorFunc = row => row.get(0).asInstanceOf[Vector]

    override def transformSchema(schema: StructType) : StructType = {
        SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT(), false)
    }

    /**
      * Fits a dataframe to the specified network configuration
      * @param dataset DataFrame
      * @return Returns an autoencoder model, which can run transformations on the vector
      */
    override def fit(dataset: DataFrame) : AutoEncoderModel = {
        val sparkdl4j = fitter(DatasetFacade.dataRows(dataset))
        new AutoEncoderModel(uid, sparkdl4j, _multiLayerConfiguration)
            .setInputCol($(inputCol))
            .setOutputCol($(outputCol))
            .setCompressedLayer($(compressedLayer))
    }

}

class AutoEncoderModel(uid: String, multiLayerNetwork: MultiLayerNetwork, multiLayerConfiguration: MultiLayerConfiguration)
    extends AutoEncoderModelWrapper[AutoEncoderModel](uid, multiLayerNetwork, multiLayerConfiguration) {

    override def udfTransformer = udf[Vector, Vector](vec => {
        val out = multiLayerNetwork.feedForwardToLayer($(compressedLayer), Nd4j.create(vec.toArray))
        val mainLayer = out.get(out.size() - 1)

        // FIXME: int cast
        val size =  mainLayer.size(1).toInt
        val values = Array.fill(size)(0.0)
        for (i <- 0 until size) {
            values(i) = mainLayer.getDouble(i.toLong)
        }
        Vectors.dense(values)
    })

    /**
      * copys an autoencoder model, including the param map
      * @param extra ParamMap
      * @return returns a copy of the autoencoder model
      */
    override def copy(extra: ParamMap) : AutoEncoderModel = {
        copyValues(new AutoEncoderModel(uid, multiLayerNetwork, multiLayerConfiguration)).setParent(parent)
    }

    /**
      * Transforms an incoming dataframe
      * @param dataFrame DataFrame
      * @return Returns a transformed dataframe.
      */
    override def transform(dataFrame: DataFrame) : DataFrame = {
        dataFrame.withColumn($(outputCol), udfTransformer(col($(inputCol))))
    }

    /**
      * Updates the schema from the new dataframe
      * @param schema StructType
      * @return Returns a struct type
      */
    override def transformSchema(schema: StructType) : StructType = {
        SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT(), false)
    }
}

object AutoEncoderModel extends AutoEncoderModelLoader