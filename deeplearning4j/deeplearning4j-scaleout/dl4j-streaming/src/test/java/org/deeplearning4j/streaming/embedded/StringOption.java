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

package org.deeplearning4j.streaming.embedded;

import scala.Option;

/**
 * Created by agibsonccc on 6/9/16.
 */
public class StringOption extends Option<String> {
    private String value;

    public StringOption(String value) {
        this.value = value;
    }

    @Override
    public boolean isEmpty() {
        return value == null || value.isEmpty();
    }

    @Override
    public String get() {
        return value;
    }

    @Override
    public Object productElement(int n) {
        return value.charAt(n);
    }

    @Override
    public int productArity() {
        return value.length();
    }

    @Override
    public boolean canEqual(Object that) {
        return that instanceof String;
    }

    @Override
    public boolean equals(Object that) {
        return that.equals(value);
    }

    @Override
    public int hashCode() {
        return value.hashCode();
    }
}
