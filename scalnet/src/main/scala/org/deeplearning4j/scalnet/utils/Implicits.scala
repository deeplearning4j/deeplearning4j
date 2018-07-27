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

package org.deeplearning4j.scalnet.utils

import scala.reflect.ClassTag

/**
  * Created by maxpumperla on 17/07/17.
  */
object Implicits {

  implicit class WithAsInstanceOfOpt(obj: AnyRef) {

    /**
      * Half type-safe cast. It uses erasure semantics (like Java casts). For example:
      *
      *  `xs: List[Int]`
      *
      *  `xs.asInstanceOfOpt[List[Int]] == xs.asInstanceOfOpt[List[Double]] == xs.asInstanceOfOpt[Seq[Int]] == Some(xs)`
      *
      *  and
      *
      *  `xs.asInstanceOfOpt[String] == xs.asInstanceOfOpt[Set[Int]] == None`
      *
      *  @return None if the cast fails or the object is `null`, `Some[B]` otherwise
      */
    def asInstanceOfOpt[B: ClassTag]: Option[B] = obj match {
      case b: B => Some(b)
      case _    => None
    }
  }

}
