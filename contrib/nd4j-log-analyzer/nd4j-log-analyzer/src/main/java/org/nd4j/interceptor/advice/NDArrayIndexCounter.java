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

package org.nd4j.interceptor.advice;

import org.nd4j.common.primitives.CounterMap;

public class NDArrayIndexCounter {

    private static CounterMap<String,String> counterMap = new CounterMap<>();


    public static int getCount(String className,String methodName) {
        return (int) counterMap.getCount(className,methodName);
    }
    public static void increment(String className,String methodName) {
        counterMap.incrementCount(className,methodName,1.0);
    }

}
