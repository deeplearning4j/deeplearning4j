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

package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

/**
 * This function is used to
 *
 * @author raver119@gmail.com
 */
public class ExportFunction<T extends SequenceElement> implements VoidFunction<T> {

    public ExportFunction(Broadcast<VocabCache<ShallowSequenceElement>> vocabCacheBroadcast,
                    @NonNull String hdfsFilePath) {

    }

    @Override
    public void call(T t) throws Exception {

    }
}
