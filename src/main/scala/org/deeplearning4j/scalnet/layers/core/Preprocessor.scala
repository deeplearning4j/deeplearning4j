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

package org.deeplearning4j.scalnet.layers.core

import org.deeplearning4j.nn.conf.InputPreProcessor

/**
  * Trait for preprocessing layers in DL4J neural networks and computational
  * graphs. Compiles out to DL4J InputPreProcessor.
  *
  * @author David Kale
  */
trait Preprocessor extends Node {
  def compile: InputPreProcessor
}
