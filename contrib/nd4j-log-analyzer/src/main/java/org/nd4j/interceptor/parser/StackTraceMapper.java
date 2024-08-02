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
package org.nd4j.interceptor.parser;

import java.util.*;

public class StackTraceMapper {
    private Map<String, String> mappedStackTraces = new HashMap<>();
    private Map<String, String> reverseMappedStackTraces = new HashMap<>();


    public List<String> getLinesOfCodeFromStackTrace(SourceCodeIndexer indexer, String[] stackTrace) {
        List<String> linesOfCode = new ArrayList<>();
        StackTraceElement[] parsedStackTrace = parseStackTrace(stackTrace);

        for (StackTraceElement element : parsedStackTrace) {
            int lineNumber = element.getLineNumber();
            String lineOfCode = indexer.getSourceCodeLine(element.getClassName(), lineNumber).getLine();
            if (lineOfCode != null) {
                linesOfCode.add(lineOfCode);
            }
        }

        return linesOfCode;
    }

    public void mapStackTraces(String[] stackTrace1, String[] stackTrace2) {
        List<String> listStackTrace1 = Arrays.asList(stackTrace1);
        List<String> listStackTrace2 = Arrays.asList(stackTrace2);
        mapStackTraces(listStackTrace1, listStackTrace2);
    }

    public void mapStackTraces(List<String> stackTrace1, List<String> stackTrace2) {
        List<StackTraceElement> parsedStackTrace1 = parseStackTrace(stackTrace1);
        List<StackTraceElement> parsedStackTrace2 = parseStackTrace(stackTrace2);

        for (StackTraceElement element1 : parsedStackTrace1) {
            for (StackTraceElement element2 : parsedStackTrace2) {
                if (element1.getMethodName().equals(element2.getMethodName()) && element1.getLineNumber() == element2.getLineNumber()) {
                    mappedStackTraces.put(element1.toString(), element2.toString());
                    reverseMappedStackTraces.put(element2.toString(), element1.toString());
                }
            }
        }
    }

    public String lookupMethod(String method) {
        return mappedStackTraces.getOrDefault(method, null);
    }

    public String reverseLookupMethod(String method) {
        return reverseMappedStackTraces.getOrDefault(method, null);
    }

    private StackTraceElement[] parseStackTrace(String[] stackTrace) {
        StackTraceElement[] parsedStackTrace = new StackTraceElement[stackTrace.length];
        for(int i = 0; i < stackTrace.length; i++) {
            parsedStackTrace[i] = parseStackTraceLine(stackTrace[i]);
        }
        return parsedStackTrace;
    }

    private List<StackTraceElement> parseStackTrace(List<String> stackTrace) {
        List<StackTraceElement> parsedStackTrace = new ArrayList<>();

        for (String trace : stackTrace) {
            StackTraceElement element = parseStackTraceLine(trace);
            if (element != null) {
                parsedStackTrace.add(element);
            }
        }

        return parsedStackTrace;
    }

    private StackTraceElement parseStackTraceLine(String stackTraceLine) {
        String[] parts = stackTraceLine.split("\\.");
        String className = String.join(".", java.util.Arrays.copyOfRange(parts, 0, parts.length - 1));
        String methodName = parts[parts.length - 1].split("\\(")[0];
        String fileName = parts[parts.length - 1].split("\\(")[1].split(":")[0];
        int lineNumber = Integer.parseInt(parts[parts.length - 1].split("\\(")[1].split(":")[1].replace(")", ""));
        return new StackTraceElement(className, methodName, fileName, lineNumber);
    }
}