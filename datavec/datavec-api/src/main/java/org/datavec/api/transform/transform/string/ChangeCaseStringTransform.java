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

package org.datavec.api.transform.transform.string;

import lombok.Data;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Change case (to, e.g, all lower case) of String column.
 *
 * @author dave@skymind.io
 */
@Data
public class ChangeCaseStringTransform extends BaseStringTransform {
    public enum CaseType {
        LOWER, UPPER
    }

    private final CaseType caseType;

    public ChangeCaseStringTransform(String column) {
        super(column);
        this.caseType = CaseType.LOWER; // default is all lower case
    }

    public ChangeCaseStringTransform(@JsonProperty("column") String column,
                    @JsonProperty("caseType") CaseType caseType) {
        super(column);
        this.caseType = caseType;
    }

    private String mapHelper(String input) {
        String result;
        switch (caseType) {
            case UPPER:
                result = input.toUpperCase();
                break;
            case LOWER:
            default:
                result = input.toLowerCase();
                break;
        }
        return result;
    }

    @Override
    public Text map(Writable writable) {
        return new Text(mapHelper(writable.toString()));
    }

    @Override
    public Object map(Object input) {
        return mapHelper(input.toString());
    }
}
