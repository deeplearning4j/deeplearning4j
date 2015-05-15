/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.apache.spark.ml.nn

import org.apache.spark.annotation.{AlphaComponent, DeveloperApi}
import org.apache.spark.ml.param._
import org.apache.spark.ml.nn._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.linalg.{VectorUDT, Vector}
import org.apache.spark.sql.types.{DataType, DoubleType, FloatType, IntegerType, StructField, StructType}
import org.apache.spark.sql.DataFrame

trait UnsupervisedLearnerParams extends Params 
  with HasFeaturesCol {
  
  protected def validateAndTransformSchema(
      schema: StructType,
      paramMap: ParamMap,
      fitting: Boolean,
      featuresDataType: DataType): StructType = {
    val map = this.paramMap ++ paramMap
    // TODO: Support casting Array[Double] and Array[Float] to Vector when FeaturesType = Vector
    checkInputColumn(schema, map(featuresCol), featuresDataType)
    schema
  }
}

/**
 * Abstract unsupervised learning algorithm.
 */
@AlphaComponent
abstract class UnsupervisedLearner[
    FeaturesType,
    Learner <: UnsupervisedLearner[FeaturesType, Learner, M],
    M <: UnsupervisedModel[FeaturesType, M]]
  extends Estimator[M] with UnsupervisedLearnerParams {
  
  /** @group setParam */
  def setFeaturesCol(value: String): Learner = set(featuresCol, value).asInstanceOf[Learner]
  
  override def fit(dataset: DataFrame, paramMap: ParamMap): M = {
    // This handles a few items such as schema validation.
    // Developers only need to implement learn().
    transformSchema(dataset.schema, paramMap, logging = true)
    val map = this.paramMap ++ paramMap
    val model = learn(dataset, map)
    Params.inheritValues(map, this, model) // copy params to model
    model
  }
  
  /**
   * :: DeveloperApi ::
   *
   * Learn a model using the given dataset and parameters.
   * Developers can implement this instead of [[fit()]] to avoid dealing with schema validation
   * and copying parameters into the model.
   *
   * @param dataset  Learning dataset
   * @param paramMap  Parameter map.  Unlike [[fit()]]'s paramMap, this paramMap has already
   *                  been combined with the embedded ParamMap.
   * @return  Fitted model
   */
  @DeveloperApi
  protected def learn(dataset: DataFrame, paramMap: ParamMap): M

  /**
   * :: DeveloperApi ::
   *
   * Returns the SQL DataType corresponding to the FeaturesType type parameter.
   *
   * This is used by [[validateAndTransformSchema()]].
   * This workaround is needed since SQL has different APIs for Scala and Java.
   *
   * The default value is VectorUDT, but it may be overridden if FeaturesType is not Vector.
   */
  @DeveloperApi
  protected def featuresDataType: DataType = new VectorUDT
  
  override def transformSchema(schema: StructType, paramMap: ParamMap): StructType = {
    validateAndTransformSchema(schema, paramMap, fitting = true, featuresDataType)
  }
}

@AlphaComponent
abstract class UnsupervisedModel[FeaturesType, M <: UnsupervisedModel[FeaturesType, M]]
  extends Model[M] with UnsupervisedLearnerParams {

  /** @group setParam */
  def setFeaturesCol(value: String): M = set(featuresCol, value).asInstanceOf[M]
  
  /**
   * :: DeveloperApi ::
   *
   * Returns the SQL DataType corresponding to the FeaturesType type parameter.
   *
   * This is used by [[validateAndTransformSchema()]].
   * This workaround is needed since SQL has different APIs for Scala and Java.
   *
   * The default value is VectorUDT, but it may be overridden if FeaturesType is not Vector.
   */
  @DeveloperApi
  protected def featuresDataType: DataType = new VectorUDT

  override def transformSchema(schema: StructType, paramMap: ParamMap): StructType = {
    validateAndTransformSchema(schema, paramMap, fitting = false, featuresDataType)
  }
  
  /**
   * Transforms dataset by reading from [[featuresCol]].
   *
   * @param dataset input dataset
   * @param paramMap additional parameters, overwrite embedded params
   * @return transformed dataset.
   */
  override def transform(dataset: DataFrame, paramMap: ParamMap): DataFrame = {

    // Check schema
    transformSchema(dataset.schema, paramMap, logging = true)
    val map = this.paramMap ++ paramMap

    // Prepare model
    val tmpModel = if (paramMap.size != 0) {
      val tmpModel = this.copy()
      Params.inheritValues(paramMap, parent, tmpModel)
      tmpModel
    } else {
      this
    }
    
    tmpModel.predict(dataset, map)
  }
  
  /**
   * :: DeveloperApi ::
   *
   * Predict for the given features.
   * This internal method is used to implement [[transform()]] and output some column(s).
   */
  @DeveloperApi
  def predict(dataset: DataFrame, paramMap: ParamMap): DataFrame 
  
  /**
   * Create a copy of the model.
   * The copy is shallow, except for the embedded paramMap, which gets a deep copy.
   */
  protected def copy(): M
}