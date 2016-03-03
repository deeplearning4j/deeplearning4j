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

package org.deeplearning4j.spark.sql.sources

import org.apache.spark.sql.{DataFrameReader, SQLContext, DataFrame}

package object iris {

  /**
   * Adds a method, `iris`, to DataFrameReader that allows reading the Iris dataset.
   */
  implicit class IrisDataReader(read: DataFrameReader) {
    def iris(filePath: String) =
      read.format(classOf[DefaultSource].getName).load(filePath)
  }

}