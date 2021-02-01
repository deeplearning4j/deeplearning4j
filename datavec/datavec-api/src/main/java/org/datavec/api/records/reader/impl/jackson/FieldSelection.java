/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.records.reader.impl.jackson;

import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class FieldSelection implements Serializable {

    public static final Writable DEFAULT_MISSING_VALUE = new Text("");

    private List<String[]> fieldPaths;
    private List<Writable> valueIfMissing;

    private FieldSelection(Builder builder) {
        this.fieldPaths = builder.fieldPaths;
        this.valueIfMissing = builder.valueIfMissing;
    }

    public List<String[]> getFieldPaths() {
        return fieldPaths;
    }

    public List<Writable> getValueIfMissing() {
        return valueIfMissing;
    }

    public int getNumFields() {
        return fieldPaths.size();
    }

    public static class Builder {

        private List<String[]> fieldPaths = new ArrayList<>();
        private List<Writable> valueIfMissing = new ArrayList<>();


        /**
         *
         * @param fieldPath    Path to the field. For example, {"a", "b", "c"} would be a field named c, in an object b,
         *                     where b is in an object a
         */
        public Builder addField(String... fieldPath) {
            return addField(DEFAULT_MISSING_VALUE, fieldPath);
        }

        public Builder addField(Writable valueIfMissing, String... fieldPath) {
            fieldPaths.add(fieldPath);
            this.valueIfMissing.add(valueIfMissing);
            return this;
        }

        public FieldSelection build() {
            return new FieldSelection(this);
        }
    }

}
