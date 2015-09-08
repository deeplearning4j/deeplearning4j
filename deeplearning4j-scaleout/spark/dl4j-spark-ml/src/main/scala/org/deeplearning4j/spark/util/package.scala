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

package org.deeplearning4j.spark

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.apache.spark.mllib.linalg.{Vectors, Vector}

package object util {
  object conversions {
    /**
     * Convert a vector to an ndarray
     * @param vector the vector
     * @return an ndarray
     */
    implicit def toINDArray(vector: Vector): INDArray = {
      Nd4j.create(Nd4j.createBuffer(vector.toArray))
    }

    /**
     * Convert an ndarray to a vector
     * @param array the array
     * @return an mllib vector
     */
    implicit def toVector(array: INDArray): Vector = {
      if (!array.isVector) {
        throw new IllegalArgumentException("implicit array must be a vector")
      }
      val ret = new Array[Double](array.length)
      for(i <- 0 to array.length - 1)
        ret(i) = array.getDouble(i)

      return Vectors.dense(ret)
    }

    implicit def toMatrix(array: INDArray): Array[Vector] = {
      if (!array.isRowVector && !array.isMatrix) {
        throw new IllegalArgumentException("implicit array must be a matrix")
      }

      val matrix = new Array[Vector](array.rows())
      for(i <- 0 to array.rows() - 1) {
        val row = array.getRow(i)
        val v = new Array[Double](row.length)
        for(j <- 0 to row.length - 1)
          v(j) = row.getDouble(j)
        matrix(i) = Vectors.dense(v)
      }

      matrix
    }
  }
}
