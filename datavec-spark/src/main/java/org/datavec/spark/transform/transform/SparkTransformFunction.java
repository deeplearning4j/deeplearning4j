/*-
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.spark.transform.transform;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.function.Function;
import org.datavec.api.transform.Transform;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Alex on 5/03/2016.
 */
@AllArgsConstructor
@Slf4j
public class SparkTransformFunction implements Function<List<Writable>, List<Writable>> {

    private final Transform transform;

    @Override
    public List<Writable> call(List<Writable> v1) throws Exception {
        if (SparkTransformExecutor.isTryCatch()) {
            try {
                return transform.map(v1);
            } catch (Exception e) {
                log.warn("Error occurred " + e + " on record " + v1);
                return new ArrayList<>();
            }
        }
        return transform.map(v1);
    }
}
