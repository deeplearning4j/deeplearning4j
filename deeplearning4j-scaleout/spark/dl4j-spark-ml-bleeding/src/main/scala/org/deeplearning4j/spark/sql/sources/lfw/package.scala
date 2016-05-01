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

package object lfw {

  /**
   * Adds a method, `lfw`, to SQLContext that allows reading the LFW dataset.
   */
  implicit class LfwContext(sqlContext: SQLContext) {
    @Deprecated
    def lfw(rootImageDirectory: String) =
      sqlContext.read.lfw(rootImageDirectory)
  }

  /**
   * Adds a method, `lfw`, to DataFrameReader that allows reading the LFW dataset.
   */
  implicit class LfwDataFrameReader(read: DataFrameReader) {
    def lfw(rootImageDirectory: String) =
      read.format(classOf[DefaultSource].getName).load(rootImageDirectory)
  }
}