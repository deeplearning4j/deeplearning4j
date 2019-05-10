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

package org.deeplearning4j.nearestneighbor.server;

import play.libs.F;
import play.mvc.Result;

import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Utility methods for Routing
 *
 * @author Alex Black
 */
public class FunctionUtil {


    public static F.Function0<Result> function0(Supplier<Result> supplier) {
        return supplier::get;
    }

    public static <T> F.Function<T, Result> function(Function<T, Result> function) {
        return function::apply;
    }

}
