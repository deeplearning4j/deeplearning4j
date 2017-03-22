/*-
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.arbiter.optimize.api.evaluation;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;

import java.io.Serializable;

/**
 * ModelEvaluator: Used to conduct additional evaluation.
 * For example, this may be classification performance on a test set or similar
 */
public interface ModelEvaluator<M, D, A> extends Serializable {
    A evaluateModel(M model, DataProvider<D> dataProvider);
}
