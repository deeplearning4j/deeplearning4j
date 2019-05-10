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

package org.datavec.spark.transform.filter;

import org.apache.spark.api.java.function.Function;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.writable.NullWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

/**
 * Created by Alex on 6/03/2016.
 */
public class FilterWritablesBySchemaFunction implements Function<Writable, Boolean> {

    private final ColumnMetaData meta;
    private final boolean keepValid; //If true: keep valid. If false: keep invalid
    private final boolean excludeMissing; //If true: remove/exclude any


    public FilterWritablesBySchemaFunction(ColumnMetaData meta, boolean keepValid) {
        this(meta, keepValid, false);
    }

    /**
     *
     * @param meta              Column meta data
     * @param keepValid         If true: keep only the valid writables. If false: keep only the invalid writables
     * @param excludeMissing    If true: don't return any missing values, regardless of keepValid setting (i.e., exclude any NullWritable or empty string values)
     */
    public FilterWritablesBySchemaFunction(ColumnMetaData meta, boolean keepValid, boolean excludeMissing) {
        this.meta = meta;
        this.keepValid = keepValid;
        this.excludeMissing = excludeMissing;
    }

    @Override
    public Boolean call(Writable v1) throws Exception {
        boolean valid = meta.isValid(v1);
        if (excludeMissing && (v1 instanceof NullWritable
                        || v1 instanceof Text && (v1.toString() == null || v1.toString().isEmpty())))
            return false; //Remove (spark)
        if (keepValid)
            return valid; //Spark: return true to keep
        else
            return !valid;
    }
}
