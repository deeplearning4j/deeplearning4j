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

package org.datavec.api.util;

import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.FloatWritable;
import org.datavec.api.writable.Writable;

import java.util.ArrayList;
import java.util.List;

/**
 * Create records from the specified input
 *
 * @author Adam Gibson
 */
public class RecordUtils {

    public static List<Writable> toRecord(double[] record) {
        List<Writable> ret = new ArrayList<>(record.length);
        for (int i = 0; i < record.length; i++)
            ret.add(new DoubleWritable(record[i]));

        return ret;
    }


    public static List<Writable> toRecord(float[] record) {
        List<Writable> ret = new ArrayList<>(record.length);
        for (int i = 0; i < record.length; i++)
            ret.add(new FloatWritable(record[i]));

        return ret;
    }

}
