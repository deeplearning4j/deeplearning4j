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

package org.nd4j.interceptor.util;

import org.nd4j.interceptor.InterceptorEnvironment;
import org.nd4j.interceptor.data.InterceptorPersistence;
import org.nd4j.interceptor.data.OpLogEvent;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.sql.*;
import java.util.*;


public class InterceptorUtils {

    static {
        try {
            InterceptorPersistence.bootstrapDatabase(InterceptorEnvironment.CURRENT_FILE_PATH);
        } catch (SQLException e) {
            throw new RuntimeException("Failed to bootstrap database", e);
        }
    }




    public static void logOpExecution(Op op) {
        if(op.opName().contains("assign")) {
            return;
        }
        if (op.opName().contains("assign")) {
            StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
            OpLogEvent opLogEvent = OpLogEvent.builder()
                    .opName(op.opName())
                    .stackTrace(getStackTrace(stackTrace))
                    .firstNonExecutionCodeLine(StackTraceCodeFinder.getFirstLineOfCode(InterceptorEnvironment.SOURCE_CODE_INDEXER_PATH,stackTrace))
                    .inputs(op.y() != null ? convertINDArrayToMap(false, op.x(), op.y()) : convertINDArrayToMap(false, op.x()))
                    .outputs(convertINDArrayToMap(false, op.z()))
                    .build();
            InterceptorPersistence.addOpLog(opLogEvent);
        } else {
            StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
            OpLogEvent opLogEvent = OpLogEvent.builder()
                    .opName(op.opName())
                    .firstNonExecutionCodeLine(StackTraceCodeFinder.getFirstLineOfCode(InterceptorEnvironment.SOURCE_CODE_INDEXER_PATH,stackTrace))
                    .stackTrace(getStackTrace(stackTrace))
                    .inputs(op.y() != null ? convertINDArrayToMap(true, op.x(), op.y()) : convertINDArrayToMap(true, op.x()))
                    .outputs(convertINDArrayToMap(false, op.z()))
                    .build();
            InterceptorPersistence.addOpLog(opLogEvent);
        }
    }

    private static Map<Integer, String> convertINDArrayToMap(boolean dup, INDArray... arrays) {
        Map<Integer, String> map = new LinkedHashMap<>();
        for (int i = 0; i < arrays.length; i++) {
            INDArray array = arrays[i];
            String arrayString = array.isView() && dup ? array.dup().toStringFull() : array.toStringFull();
            map.put(i, arrayString);
        }
        return map;
    }

    public static void logCustomOpExecution(CustomOp op) {
        if(op.opName().contains("assign")) {
            return;
        }
        StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
        OpLogEvent opLogEvent = OpLogEvent.builder()
                .firstNonExecutionCodeLine(StackTraceCodeFinder.getFirstLineOfCode(InterceptorEnvironment.SOURCE_CODE_INDEXER_PATH,stackTrace))
                .inputs(convertINDArrayToMap(!op.opName().contains("assign"), op.inputArguments().toArray(new INDArray[0])))
                .outputs(convertINDArrayToMap(!op.opName().contains("assign"), op.outputArguments().toArray(new INDArray[0])))
                .opName(op.opName())
                .stackTrace(getStackTrace())
                .build();

        InterceptorPersistence.addOpLog(opLogEvent);
    }

    public static String getStackTrace(StackTraceElement[] stackTrace) {
        StringBuilder sb = new StringBuilder();
        for (StackTraceElement element : stackTrace) {
            sb.append(element.toString()).append(System.lineSeparator());
        }
        return sb.toString();
    }


    public static String getStackTrace() {
      return getStackTrace(Thread.currentThread().getStackTrace());
    }

}
