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

package org.deeplearning4j.models.word2vec.wordstore;

import lombok.Data;
import lombok.NonNull;

/**
 * Huffman tree node info, needed for w2v calculations.
 * Used only in StandaloneWord2Vec internals.
 *
 * @author raver119@gmail.com
 */
@Data
public class HuffmanNode {
    @NonNull
    private byte[] code;
    @NonNull
    private int[] point;
    private int idx;
    private byte length;

    public HuffmanNode() {

    }

    public HuffmanNode(byte[] code, int[] point, int index, byte length) {
        this.code = code;
        this.point = point;
        this.idx = index;
        this.length = length;
    }
}
