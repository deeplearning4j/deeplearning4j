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

package org.deeplearning4j.spark.util.data.validation;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.spark.util.data.ValidationResult;

public class ValidationResultReduceFn implements Function2<ValidationResult, ValidationResult, ValidationResult> {
    @Override
    public ValidationResult call(ValidationResult v1, ValidationResult v2) throws Exception {
        return v1.add(v2);
    }
}
