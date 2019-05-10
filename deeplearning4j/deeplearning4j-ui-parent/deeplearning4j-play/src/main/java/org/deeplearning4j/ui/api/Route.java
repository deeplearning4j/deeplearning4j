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

package org.deeplearning4j.ui.api;

import lombok.AllArgsConstructor;
import lombok.Data;
import play.mvc.Result;

import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * A Route specifies an endpoint that can be queried in the UI - along with how it should be handled
 *
 * @author Alex Black
 */
@Data
@AllArgsConstructor
public class Route {
    private final String route;
    private final HttpMethod httpMethod;
    private final FunctionType functionType;
    private final Supplier<Result> supplier;
    private final Function<String, Result> function;
    private final BiFunction<String, String, Result> function2;

    public Route(String route, HttpMethod method, FunctionType functionType, Supplier<Result> supplier) {
        this(route, method, functionType, supplier, null, null);
    }

    public Route(String route, HttpMethod method, FunctionType functionType, Function<String, Result> function) {
        this(route, method, functionType, null, function, null);
    }

    public Route(String route, HttpMethod method, FunctionType functionType,
                    BiFunction<String, String, Result> function) {
        this(route, method, functionType, null, null, function);
    }
}
