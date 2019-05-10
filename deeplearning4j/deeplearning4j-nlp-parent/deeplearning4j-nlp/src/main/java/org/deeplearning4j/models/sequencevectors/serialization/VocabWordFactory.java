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

package org.deeplearning4j.models.sequencevectors.serialization;

import org.deeplearning4j.models.sequencevectors.interfaces.SequenceElementFactory;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;

/**
 * @author raver119@gmail.com
 */
public class VocabWordFactory implements SequenceElementFactory<VocabWord> {

    /**
     * This method builds object from provided JSON
     *
     * @param json JSON for restored object
     * @return restored object
     */
    @Override
    public VocabWord deserialize(String json) {
        ObjectMapper mapper = SequenceElement.mapper();
        try {
            VocabWord ret = mapper.readValue(json, VocabWord.class);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method serializaes object  into JSON string
     *
     * @param element
     * @return
     */
    @Override
    public String serialize(VocabWord element) {
        return element.toJSON();
    }
}
