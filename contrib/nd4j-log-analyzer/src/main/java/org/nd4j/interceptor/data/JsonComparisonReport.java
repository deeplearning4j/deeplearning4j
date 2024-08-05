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
package org.nd4j.interceptor.data;
import org.json.JSONTokener;
import org.nd4j.interceptor.InterceptorEnvironment;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;


public class JsonComparisonReport {


    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Usage: java JsonComparisonReport <directory1> <directory2>");
            System.exit(1);
        }

        String directory1 = args[0];
        String directory2 = args[1];
        for(double epsilon : InterceptorEnvironment.EPSILONS) {
            Map<String,OpDifference> differences = compareDirectories(directory1, directory2,epsilon);
            generateReport(differences,epsilon);
        }

        List<OpLogEvent> orderedEvents1 = orderedEvents(new File(directory1));
        List<OpLogEvent> orderedEvents2 = orderedEvents(new File(directory2));
        try {
            InterceptorEnvironment.mapper.writeValue(new File("first_in_order.json"), orderedEvents1);
            InterceptorEnvironment.mapper.writeValue(new File("second_in_order.json"), orderedEvents2);
        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    private static Map<String,OpDifference> compareDirectories(String directory1, String directory2,double epsilon) {
        Map<String,OpDifference> differences = new HashMap<>();
        File dir1 = new File(directory1);
        File dir2 = new File(directory2);

        File[] files1 = dir1.listFiles((dir, name) -> name.endsWith(".json"));
        File[] files2 = dir2.listFiles((dir, name) -> name.endsWith(".json"));

        if (files1 != null && files2 != null) {
            for (File file1 : files1) {
                if(file1.getName().contains("div_scalar")) {
                    continue;
                }
                String fileName = file1.getName();
                File file2 = new File(dir2, fileName);

                if (file2.exists()) {
                    try {
                        System.out.println("Processing files: " + file1.getName() + " and " + file2.getName());
                        JSONObject jsonObject = new JSONObject(new JSONTokener(new FileReader(file1)));
                        JSONObject jsonObject2 = new JSONObject(new JSONTokener(new FileReader(file2)));

                        SourceCodeOpEvent eventsGrouped =  convertJsonToSourceCodeOpEvent(jsonObject);
                        SourceCodeOpEvent eventsGrouped2 = convertJsonToSourceCodeOpEvent(jsonObject2);
                        Map<String, OpDifference> opLogDifferences = compareOpLogArrays(eventsGrouped.getOpLogEvents(), eventsGrouped2.getOpLogEvents(),epsilon);
                        differences.putAll(opLogDifferences);
                    } catch (IOException | JSONException e) {
                        e.printStackTrace();
                    }
                }
            }
        }

        return differences;
    }

    private static SourceCodeOpEvent convertJsonToSourceCodeOpEvent(JSONObject jsonObject) {
        Map<String, List<OpLogEvent>> opLogEvents = new HashMap<>();
        jsonObject = jsonObject.getJSONObject("opLogEvents");
        // Iterate over the keys in the JSON object
        for (String key : jsonObject.keySet()) {
            // Get the JSONArray corresponding to the key
            JSONArray jsonArray = jsonObject.getJSONArray(key);
            List<OpLogEvent> opLogEventList = new ArrayList<>();

            // Iterate over the elements in the JSONArray
            for (int i = 0; i < jsonArray.length(); i++) {
                // Get the JSONObject representing an OpLogEvent
                JSONObject opLogEventJson = jsonArray.getJSONObject(i);

                // Convert the JSONObject to an OpLogEvent
                OpLogEvent opLogEvent = convertToOpLogEvent(opLogEventJson);

                // Add the OpLogEvent to the list
                opLogEventList.add(opLogEvent);
            }

            // Add the list of OpLogEvents to the map with the corresponding key
            opLogEvents.put(key, opLogEventList);
        }

        // Create and return a new SourceCodeOpEvent with the opLogEvents map
        return SourceCodeOpEvent.builder()
                .opLogEvents(opLogEvents)
                .build();
    }


    private static List<OpLogEvent> orderedEvents(File directory) {
        List<OpLogEvent> orderedEvents = new ArrayList<>();
        File[] files = directory.listFiles((dir, name) -> name.endsWith(".json"));
        if (files != null) {
            for (File file : files) {
                try {
                    JSONObject jsonObject = new JSONObject(new JSONTokener(new FileReader(file)));
                    jsonObject = jsonObject.getJSONObject("opLogEvents");
                    for (String key : jsonObject.keySet()) {
                        JSONArray jsonArray = jsonObject.getJSONArray(key);
                        for (int i = 0; i < jsonArray.length(); i++) {
                            JSONObject opLogEventJson = jsonArray.getJSONObject(i);
                            OpLogEvent opLogEvent = convertToOpLogEvent(opLogEventJson);
                            orderedEvents.add(opLogEvent);
                        }
                    }
                } catch (IOException | JSONException e) {
                    e.printStackTrace();
                }
            }
        }

        Collections.sort(orderedEvents, Comparator.comparingLong(OpLogEvent::getEventId));

        return orderedEvents;
    }

    private static OpLogEvent convertToOpLogEvent(JSONObject jsonObject) {
        String opName = jsonObject.getString("opName");
        JSONObject inputsObject = jsonObject.getJSONObject("inputs");
        JSONObject outputsObject = jsonObject.getJSONObject("outputs");
        String stackTrace = jsonObject.getString("stackTrace");

        Map<Integer, String> inputs = decodeInputsOutputs(inputsObject);
        Map<Integer, String> outputs = decodeInputsOutputs(outputsObject);

        return OpLogEvent.builder()
                .firstNonExecutionCodeLine(jsonObject.getString("firstNonExecutionCodeLine"))
                .opName(opName)
                .inputs(inputs)
                .outputs(outputs)
                .eventId(jsonObject.getLong("eventId"))
                .stackTrace(stackTrace)
                .build();
    }

    private static Map<Integer, String> decodeInputsOutputs(JSONObject jsonObject) {
        Map<Integer, String> result = new HashMap<>();

        for (String key : jsonObject.keySet()) {
            int index = Integer.parseInt(key);
            String value = jsonObject.getString(key);
            result.put(index, value);
        }

        return result;
    }

    private static Map<String,OpDifference> compareOpLogArrays(Map<String,List<OpLogEvent>> jsonArray1,  Map<String,List<OpLogEvent>> jsonArray2,double epsilon) {
        Map<String,OpDifference> differences = new HashMap<>();
        for (String key : jsonArray1.keySet()) {
            List<OpLogEvent> opLogEvents1 = jsonArray1.get(key);
            List<OpLogEvent> opLogEvents2 = jsonArray2.get(key);
            if(opLogEvents1 == null || opLogEvents2 == null)
                continue;
            int minEventSize = Math.min(opLogEvents1.size(), opLogEvents2.size());
            if (opLogEvents2 != null) {
                for (int i = 0; i < minEventSize; i++) {
                    OpLogEvent opLogEvent1 = opLogEvents1.get(i);
                    OpLogEvent opLogEvent2 = opLogEvents2.get(i);
                    Map<Integer,String> inputs = opLogEvent1.getInputs();
                    Map<Integer,String> outputs = opLogEvent1.getOutputs();

                    Map<Integer,String> inputs2 = opLogEvent2.getInputs();
                    Map<Integer,String> outputs2 = opLogEvent2.getOutputs();
                    for(int j = 0; j < inputs.size(); j++) {
                        if(inputs.get(j).contains("assign")) {
                            continue;
                        }
                        JSONArray jsonArray = new JSONArray(inputs.get(j));
                        JSONArray jsonArray3 = new JSONArray(inputs2.get(j));
                        JSONComparisonResult result = compareJSONArraysWithEpsilon(jsonArray, jsonArray3, epsilon);
                        if(!result.isSame()) {
                            OpDifference opDifference = OpDifference.builder()
                                    .opLog1(opLogEvent1)
                                    .opLog2(opLogEvent2)
                                    .differenceType("inputs")
                                    .differenceValue1(String.valueOf(result.getFirstValue()))
                                    .differenceValue2(String.valueOf(result.getSecondValue()))
                                    .opDifference(j)
                                    .build();
                            differences.put(key, opDifference);
                            break;
                        }
                    }

                    for(int j = 0; j < outputs.size(); j++) {
                        if(inputs.get(j).contains("assign")) {
                            continue;
                        }

                        Object cast = outputs.get(j);
                        if(cast instanceof Number) {
                            cast = new double[] {
                                    ((Number) cast).doubleValue()
                            };
                        } else if(cast instanceof String) {
                            //if string matches a single double between []

                            if(cast.toString().matches("-*\\d+\\.\\d+")) {
                                cast = new JSONArray(new double[] {
                                        Double.parseDouble((String) cast)
                                });

                            } else {
                                cast = new JSONArray(cast.toString());
                            }


                        }

                        Object cast2 = outputs2.get(j);
                        if(cast2 instanceof Number) {
                            cast2 = new double[] {
                                    ((Number) cast2).doubleValue()
                            };
                        } else if(cast2 instanceof String) {
                            //if string matches a single double between []

                            if(cast2.toString().matches("-*\\d+\\.\\d+")) {
                                cast2 = new JSONArray(new double[] {
                                        Double.parseDouble((String) cast2)
                                });

                            } else {
                                cast2 = new JSONArray(cast2.toString());
                            }
                        }

                        JSONArray casted1 = (JSONArray) cast;
                        JSONArray casted2 = (JSONArray) cast2;

                        JSONComparisonResult result = compareJSONArraysWithEpsilon(casted1, casted2, epsilon);
                        if(!result.isSame()) {
                            OpDifference opDifference = OpDifference.builder()
                                    .opLog1(opLogEvent1)
                                    .opLog2(opLogEvent2)
                                    .differenceType("outputs")
                                    .differenceValue1(String.valueOf(result.getFirstValue()))
                                    .differenceValue2(String.valueOf(result.getSecondValue()))
                                    .opDifference(result.getIndex())
                                    .build();
                            differences.put(key, opDifference);
                            break;
                        }
                    }

                }
            }
        }
        return differences;
    }


    private static void generateReport(Map<String,OpDifference> differences,double epsilon) {
        String reportFile = "comparison_report_" + epsilon + ".json";
        String earliestDifferenceFile = "earliest_difference_" + epsilon + ".json";
        String firstInOrderFile = "first_in_order_" + epsilon + ".json";
        String secondInOrderFile = "second_in_order_" + epsilon + ".json";
        Map<String,OpDifference> filteredDifferences = filterDifferencesByEpsilon(differences, epsilon);

        try {
            InterceptorEnvironment.mapper.writeValue(new File(reportFile), filteredDifferences);
            InterceptorEnvironment.mapper.writeValue(new File(earliestDifferenceFile), OpDifference.earliestDifference(filteredDifferences));

            System.out.println("Comparison report for epsilon " + epsilon + " saved to: " + reportFile);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private static Map<String,OpDifference> filterDifferencesByEpsilon(Map<String,OpDifference> differences, double epsilon) {
        Map<String,OpDifference> filteredDifferences = new HashMap<>();

        for (Map.Entry<String,OpDifference> difference : differences.entrySet()) {
            if (isDifferentWithEpsilon(difference.getValue().getOpLog1(), difference.getValue().getOpLog2(), epsilon)) {
                filteredDifferences.put(difference.getKey(),difference.getValue());
            }
        }

        return filteredDifferences;
    }

    private static Map<String,String> convertIntMap(Map<Integer,String> map) {
        Map<String,String> newMap = new HashMap<>();
        for (Map.Entry<Integer,String> entry : map.entrySet()) {
            newMap.put(entry.getKey().toString(), entry.getValue());
        }
        return newMap;
    }


    private static boolean isDifferentWithEpsilon(OpLogEvent left, OpLogEvent right, double epsilon) {
        JSONObject leftInputs = new JSONObject(convertIntMap(left.getInputs()));
        JSONObject rightInputs = new JSONObject(convertIntMap(right.getInputs()));
        JSONObject leftOutputs = new JSONObject(convertIntMap(left.getOutputs()));
        JSONObject rightOutputs = new JSONObject(convertIntMap(right.getOutputs()));

        return !compareJSONArraysWithEpsilon(leftInputs, rightInputs, epsilon).isSame()
                || !compareJSONArraysWithEpsilon(leftOutputs, rightOutputs, epsilon).isSame();
    }


    private static JSONComparisonResult compareJSONArraysWithEpsilon(JSONArray jsonArray1, JSONArray jsonArray2, double epsilon) {
        if (jsonArray1.length() != jsonArray2.length()) {
            return JSONComparisonResult.noDifference();
        }

        for (int i = 0; i < jsonArray1.length(); i++) {
            Object value1 = jsonArray1.get(i);
            Object value2 = jsonArray2.get(i);
            if(value1 instanceof JSONArray) {
                JSONComparisonResult result = compareJSONArraysWithEpsilon((JSONArray) value1,(JSONArray) value2,epsilon);
                if(!result.isSame()) {
                    return result;
                }

                continue;
            }


            if (Math.abs(((Number) value1).doubleValue() - ((Number) value2).doubleValue()) > epsilon) {
                return JSONComparisonResult.builder()
                        .same(false)
                        .firstValue(((Number) value1).doubleValue())
                        .secondValue(((Number) value2).doubleValue())
                        .build();
            }
        }

        return JSONComparisonResult.noDifference();
    }


    private static JSONComparisonResult compareJSONArraysWithEpsilon(JSONObject jsonArray1, JSONObject jsonArray2, double epsilon) {
        if (jsonArray1.length() != jsonArray2.length()) {
            return JSONComparisonResult.noDifference();
        }

        for (int i = 0; i < jsonArray1.length(); i++) {
            Object cast1 = jsonArray1.get(String.valueOf(i));
            if(cast1 instanceof String) {
                cast1 = new JSONArray(cast1.toString());
            }

            Object cast2 = jsonArray2.get(String.valueOf(i));
            if(cast2 instanceof String) {
                cast2 = new JSONArray(cast2.toString());
            }
            JSONArray value1 = (JSONArray) cast1;
            JSONArray value2 = (JSONArray) cast2;
            JSONComparisonResult result = compareJSONArraysWithEpsilon(value1,value2,epsilon);
            if(!result.isSame()) {
                return result;
            }
        }

        return JSONComparisonResult.noDifference();
    }


}