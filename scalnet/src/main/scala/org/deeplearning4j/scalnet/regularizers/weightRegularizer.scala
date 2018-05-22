/*
 * Copyright 2016 Skymind
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.deeplearning4j.scalnet.regularizers

/**
  * Weight regularizers.
  *
  * @author David Kale
  */
sealed class WeightRegularizer(val l1: Double = Double.NaN, val l2: Double = Double.NaN)

case class NoRegularizer() extends WeightRegularizer()
case class L1(l: Double = 0.01) extends WeightRegularizer(l1 = l)
case class L2(l: Double = 0.01) extends WeightRegularizer(l2 = l)
case class L1L2(override val l1: Double = 0.01, override val l2: Double = 0.01) extends WeightRegularizer
