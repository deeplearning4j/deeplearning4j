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

package org.deeplearning4j.spark.models.sequencevectors.export;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Arrays;
import java.util.regex.Pattern;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ExportContainer<T extends SequenceElement> implements Serializable {
    private T element;
    private INDArray array;

    protected static final Pattern pattern0 = Pattern.compile("(\\[|\\])");
    protected static final Pattern pattern1 = Pattern.compile("(\\,|\\.|\\;)+\\s");

    // TODO: implement B64 optional compression here?
    @Override
    public String toString() {
        // TODO: we need proper string cleansing here

        String ars = Arrays.toString(array.data().asFloat());
        ars = pattern0.matcher(ars).replaceAll("").trim();
        ars = pattern1.matcher(ars).replaceAll(" ").trim();

        return element.getLabel().trim() + " " + ars;
    }
}
