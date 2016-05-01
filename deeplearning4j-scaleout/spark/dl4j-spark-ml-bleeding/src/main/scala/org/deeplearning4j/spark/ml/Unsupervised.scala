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

package org.deeplearning4j.spark.ml

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param._
import org.deeplearning4j.spark.ml.param.shared._
import org.deeplearning4j.spark.ml.util.SchemaUtils
import org.apache.spark.ml.{PipelineStage, Estimator, Model}
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DataType, StructType}
import org.deeplearning4j.spark.sql.types.VectorUDT

/**
 * :: DeveloperApi ::
 *
 * Trait for parameters for unsupervised learning.
 */
trait UnsupervisedLearnerParams extends Params 
  with HasFeaturesCol {

  /**
   * Validates and transforms the input schema.
   * @param schema input schema
   * @param fitting whether this is in fitting
   * @param featuresDataType  SQL DataType for FeaturesType.
   *                          E.g., [[org.apache.spark.mllib.linalg.VectorUDT]] for vector features.
   * @return output schema
   */
  protected def validateAndTransformSchema(
      schema: StructType,
      fitting: Boolean,
      featuresDataType: DataType): StructType = {
    // TODO: Support casting Array[Double] and Array[Float] to Vector when FeaturesType = Vector
    SchemaUtils.checkColumnType(schema, $(featuresCol), featuresDataType)
    schema
  }
}

/**
 * Abstract unsupervised learning algorithm.
 *
 * @author Eron Wright
 */
@DeveloperApi
abstract class UnsupervisedLearner[
    FeaturesType,
    Learner <: UnsupervisedLearner[FeaturesType, Learner, M],
    M <: UnsupervisedModel[FeaturesType, M]]
  extends Estimator[M] with UnsupervisedLearnerParams {
  
  /** @group setParam */
  def setFeaturesCol(value: String): Learner = set(featuresCol, value).asInstanceOf[Learner]
  setDefault(featuresCol -> "features")

  override def fit(dataset: DataFrame): M = {
    // This handles a few items such as schema validation.
    // Developers only need to implement learn().
    transformSchema(dataset.schema, logging = true)
    copyValues(learn(dataset).setParent(this))
  }

  override def copy(extra: ParamMap): Learner = {
    this.asInstanceOf[PipelineStage].copy(extra).asInstanceOf[Learner]
  }

  /**
   * Learn a model using the given dataset.
   * Developers can implement this instead of [[fit()]] to avoid dealing with schema validation.
   *
   * @param dataset  Learning dataset
   * @return  Fitted model
   */
  protected def learn(dataset: DataFrame): M

  /**
   * Returns the SQL DataType corresponding to the FeaturesType type parameter.
   *
   * This is used by [[validateAndTransformSchema()]].
   * This workaround is needed since SQL has different APIs for Scala and Java.
   *
   * The default value is VectorUDT, but it may be overridden if FeaturesType is not Vector.
   */
  protected def featuresDataType: DataType = VectorUDT()
  
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, fitting = true, featuresDataType)
  }
}

/**
 * :: DeveloperApi ::
 * Abstraction for a model for unsupervised learning tasks.
 *
 * @tparam FeaturesType  Type of features.
 *                       E.g., [[org.apache.spark.mllib.linalg.VectorUDT]] for vector features.
 * @tparam M  Specialization of [[Model]].  If you subclass this type, use this type
 *            parameter to specify the concrete type for the corresponding model.
 *
 * @author Eron Wright
 */
@DeveloperApi
abstract class UnsupervisedModel[FeaturesType, M <: UnsupervisedModel[FeaturesType, M]]
  extends Model[M] with UnsupervisedLearnerParams {

  /** @group setParam */
  def setFeaturesCol(value: String): M = set(featuresCol, value).asInstanceOf[M]
  
  /**
   * Returns the SQL DataType corresponding to the FeaturesType type parameter.
   *
   * This is used by [[validateAndTransformSchema()]].
   * This workaround is needed since SQL has different APIs for Scala and Java.
   *
   * The default value is VectorUDT, but it may be overridden if FeaturesType is not Vector.
   */
  protected def featuresDataType: DataType = VectorUDT()

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, fitting = false, featuresDataType)
  }
  
  /**
   * Transforms dataset by reading from [[featuresCol]].
   *
   * @param dataset input dataset
   * @return transformed dataset.
   */
  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    predict(dataset)
  }
  
  /**
   * Predict for the given features.
   * This internal method is used to implement [[transform()]] and output some column(s).
   */
  def predict(dataset: DataFrame): DataFrame
}