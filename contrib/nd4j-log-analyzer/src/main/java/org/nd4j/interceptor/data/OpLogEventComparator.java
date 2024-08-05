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

import org.json.JSONArray;
import org.json.JSONException;
import org.nd4j.common.primitives.Pair;
import org.nd4j.interceptor.InterceptorEnvironment;

import java.sql.*;
import java.util.*;

public class OpLogEventComparator {

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.out.println("Please provide two database file paths and an epsilon value as arguments.");
            return;
        }

        String jdbcUrl1 = "jdbc:h2:file:" + args[0];
        String jdbcUrl2 = "jdbc:h2:file:" + args[1];
        compareLinesBySide(jdbcUrl1, jdbcUrl2,1e-12);
        double epsilon = Double.parseDouble(args[2]);

        try {
            Map<String, List<OpDifference>> differences = findDifferences(jdbcUrl1, jdbcUrl2, epsilon);

            if (!differences.isEmpty()) {
                System.out.println("Found differences:");
                for (Map.Entry<String, List<OpDifference>> entry : differences.entrySet()) {
                    System.out.println("Line of code: " + entry.getKey());
                    for (OpDifference diff : entry.getValue()) {
                        System.out.println("  Difference Type: " + diff.getDifferenceType());
                        System.out.println("  Op Name: " + (diff.getOpLog1() != null ? diff.getOpLog1().getOpName() : diff.getOpLog2().getOpName()));
                        System.out.println("  Difference Value 1: " + diff.getDifferenceValue1());
                        System.out.println("  Difference Value 2: " + diff.getDifferenceValue2());
                        System.out.println("  Op Difference: " + diff.getOpDifference());
                        System.out.println();
                    }
                }
            } else {
                System.out.println("No differences found for the same inputs within the specified epsilon.");
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public static Map<String, List<OpDifference>> findDifferences(String jdbcUrl1, String jdbcUrl2, double epsilon) throws SQLException {
        String query = "SELECT id, sourceCodeLine, opName, inputs, outputs, stackTrace FROM OpLogEvent ORDER BY id";
        Map<String, List<OpLogEvent>> events1 = new LinkedHashMap<>();
        Map<String, List<OpLogEvent>> events2 = new LinkedHashMap<>();

        try (Connection conn1 = DriverManager.getConnection(jdbcUrl1, InterceptorEnvironment.USER, InterceptorEnvironment.PASSWORD);
             Connection conn2 = DriverManager.getConnection(jdbcUrl2, InterceptorEnvironment.USER, InterceptorEnvironment.PASSWORD);
             Statement stmt1 = conn1.createStatement();
             Statement stmt2 = conn2.createStatement();
             ResultSet rs1 = stmt1.executeQuery(query);
             ResultSet rs2 = stmt2.executeQuery(query)) {

            processResultSet(rs1, events1);
            processResultSet(rs2, events2);
        }

        return compareOpLogArrays(events1, events2, epsilon);
    }

    private static void processResultSet(ResultSet rs, Map<String, List<OpLogEvent>> events) throws SQLException {
        while (rs.next()) {
            OpLogEvent event = createOpLogEvent(rs);
            String sourceLine = event.getFirstNonExecutionCodeLine();
            events.computeIfAbsent(sourceLine, k -> new ArrayList<>()).add(event);
        }
    }

    private static OpLogEvent createOpLogEvent(ResultSet rs) throws SQLException {
        return OpLogEvent.builder()
                .eventId(rs.getLong("id"))
                .firstNonExecutionCodeLine(rs.getString("sourceCodeLine"))
                .opName(rs.getString("opName"))
                .inputs(convertResult(rs.getArray("inputs").getArray()))
                .outputs(convertResult(rs.getArray("outputs").getArray()))
                .stackTrace(rs.getString("stackTrace"))
                .build();
    }

    private static Map<Integer, String> convertResult(Object input) {
        Object[] inputArr = (Object[]) input;
        Map<Integer, String> ret = new LinkedHashMap<>();
        for (int i = 0; i < inputArr.length; i++) {
            ret.put(i, inputArr[i].toString());
        }
        return ret;
    }

    private static Map<String, List<OpDifference>> compareOpLogArrays(Map<String, List<OpLogEvent>> events1, Map<String, List<OpLogEvent>> events2, double epsilon) {
        Map<String, List<OpDifference>> differences = new LinkedHashMap<>();
        Map<String, OpDifference> earliestDifferences = new LinkedHashMap<>();
        Map<String, OpDifference> earliestSignificantDifferences = new LinkedHashMap<>();

        for (String line : events1.keySet()) {
            List<OpLogEvent> opLogEvents1 = events1.get(line);
            List<OpLogEvent> opLogEvents2 = events2.getOrDefault(line, new ArrayList<>());

            List<OpDifference> lineDifferences = new ArrayList<>();
            OpDifference earliestDifference = null;
            OpDifference earliestSignificantDifference = null;

            int minSize = Math.min(opLogEvents1.size(), opLogEvents2.size());
            for (int i = 0; i < minSize; i++) {
                OpLogEvent opLogEvent1 = opLogEvents1.get(i);
                OpLogEvent opLogEvent2 = opLogEvents2.get(i);

                // Compare inputs
                OpDifference inputDifference = compareInputs(opLogEvent1.getInputs(), opLogEvent2.getInputs(), epsilon, opLogEvent1, opLogEvent2);
                if (isValidDifference(inputDifference) && isSignificantDifference(inputDifference, epsilon)) {
                    lineDifferences.add(inputDifference);
                    earliestDifference = updateEarliestDifference(earliestDifference, inputDifference);
                    earliestSignificantDifference = updateEarliestDifference(earliestSignificantDifference, inputDifference);
                }

                // Compare outputs
                OpDifference outputDifference = compareOutputs(opLogEvent1.getOutputs(), opLogEvent2.getOutputs(), epsilon, opLogEvent1, opLogEvent2);
                if (isValidDifference(outputDifference) && isSignificantDifference(outputDifference, epsilon)) {
                    lineDifferences.add(outputDifference);
                    earliestDifference = updateEarliestDifference(earliestDifference, outputDifference);
                    earliestSignificantDifference = updateEarliestDifference(earliestSignificantDifference, outputDifference);
                }
            }

            if (!lineDifferences.isEmpty()) {
                differences.put(line, lineDifferences);
            }
            if (earliestDifference != null) {
                earliestDifferences.put(line, earliestDifference);
            }
            if (earliestSignificantDifference != null) {
                earliestSignificantDifferences.put(line, earliestSignificantDifference);
            }
        }

        // Check for lines in events2 that are not in events1
        for (String line : events2.keySet()) {
            if (!events1.containsKey(line)) {
                List<OpLogEvent> opLogEvents2 = events2.get(line);
                OpDifference missingLineDifference = OpDifference.builder()
                        .opLog1(null)
                        .opLog2(opLogEvents2.get(0))
                        .differenceType("missing_line")
                        .differenceValue1("null")
                        .differenceValue2(line)
                        .opDifference(opLogEvents2.size())
                        .build();
                if (isValidDifference(missingLineDifference) && isSignificantDifference(missingLineDifference, epsilon)) {
                    differences.put(line, Collections.singletonList(missingLineDifference));
                    earliestDifferences.put(line, missingLineDifference);
                    earliestSignificantDifferences.put(line, missingLineDifference);
                }
            }
        }

        // Remove any invalid or insignificant elements from the final differences result
        differences.entrySet().removeIf(entry -> entry.getValue().isEmpty());

        // Print out the earliest difference for each line
        System.out.println("Earliest differences per line of code:");
        printDifferences(earliestDifferences);

        // Create a final sorted list of lines of code with the earliest difference
        List<Map.Entry<String, OpDifference>> sortedEarliestDifferences = sortDifferences(earliestDifferences);

        // Print the sorted list of earliest differences
        System.out.println("\nSorted list of lines of code with the earliest difference:");
        printSortedDifferences(sortedEarliestDifferences);

        // Create and print a sorted list of significant differences
        List<Map.Entry<String, OpDifference>> sortedSignificantDifferences = sortDifferences(earliestSignificantDifferences);
        System.out.println("\nSorted list of lines of code with significant differences:");
        printSortedDifferences(sortedSignificantDifferences);

        return differences;
    }


    private static Map<String, List<OpLogEvent>> loadEvents(String jdbcUrl) throws SQLException {
        Map<String, List<OpLogEvent>> events = new HashMap<>();
        String query = "SELECT id, sourceCodeLine, opName, inputs, outputs, stackTrace FROM OpLogEvent ORDER BY id";

        try (Connection conn = DriverManager.getConnection(jdbcUrl, InterceptorEnvironment.USER, InterceptorEnvironment.PASSWORD);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(query)) {
            processResultSet(rs, events);
        }

        return events;
    }

    private static class LineComparison {
        String line;
        List<OpLogEvent> events1;
        List<OpLogEvent> events2;
        String difference;

        LineComparison(String line, List<OpLogEvent> events1, List<OpLogEvent> events2, String difference) {
            this.line = line;
            this.events1 = events1;
            this.events2 = events2;
            this.difference = difference;
        }

        long getEarliestEventTime() {
            long time1 = events1.isEmpty() ? Long.MAX_VALUE : events1.get(0).getEventId();
            long time2 = events2.isEmpty() ? Long.MAX_VALUE : events2.get(0).getEventId();
            return Math.min(time1, time2);
        }
    }

    public static void compareLinesBySide(String jdbcUrl1, String jdbcUrl2, double epsilon) throws SQLException {
        Map<String, List<OpLogEvent>> events1 = loadAndNormalizeEvents(jdbcUrl1);
        Map<String, List<OpLogEvent>> events2 = loadAndNormalizeEvents(jdbcUrl2);

        Set<String> allLines = new LinkedHashSet<>(events1.keySet());
        allLines.addAll(events2.keySet());

        List<LineComparison> comparisons = new ArrayList<>();
        for (String line : allLines) {
            List<OpLogEvent> lineEvents1 = events1.getOrDefault(line, Collections.emptyList());
            List<OpLogEvent> lineEvents2 = events2.getOrDefault(line, Collections.emptyList());
            Pair<String,Integer> difference = findSignificantDifference(lineEvents1, lineEvents2, epsilon);
            if (difference != null) {
                comparisons.add(new LineComparison(line, lineEvents1, lineEvents2, difference.getKey()));
            }
        }

        if (comparisons.isEmpty()) {
            System.out.println("No significant differences found.");
            return;
        }

        comparisons.sort(Comparator.comparing(LineComparison::getEarliestEventTime));

        System.out.println("Side-by-side comparison of lines with significant differences (sorted by earliest event time):");
        System.out.println("----------------------------------------------------");
        System.out.printf("%-50s | %-50s | %-30s%n", "Database 1", "Database 2", "Difference");
        System.out.println("----------------------------------------------------");

        for (LineComparison comparison : comparisons) {
            String summary1 = summarizeEvents(comparison.events1);
            String summary2 = summarizeEvents(comparison.events2);

            System.out.printf("%-50s | %-50s | %-30s%n", summary1, summary2, comparison.difference);
            System.out.println("Line: " + comparison.line);
            System.out.println("----------------------------------------------------");
        }
    }

    private static String summarizeEvents(List<OpLogEvent> events) {
        if (events.isEmpty()) {
            return "<No events>";
        }

        int count = events.size();
        String firstOpName = events.get(0).getOpName();
        long earliestEventId = events.get(0).getEventId();
        return String.format("%d events, first: %s (ID: %d)", count, firstOpName, earliestEventId);
    }

    private static Pair<String,Integer> findSignificantDifference(List<OpLogEvent> events1, List<OpLogEvent> events2, double epsilon) {
        int minSize = Math.min(events1.size(), events2.size());

        for (int i = 0; i < minSize; i++) {
            OpLogEvent e1 = events1.get(i);
            OpLogEvent e2 = events2.get(i);


            String inputDiff = compareWithEpsilon(e1.getInputs(), e2.getInputs(), epsilon);
            if (inputDiff != null) {
                return Pair.of("Inputs differ: " + inputDiff,i);
            }

            String outputDiff = compareWithEpsilon(e1.getOutputs(), e2.getOutputs(), epsilon);
            if (outputDiff != null) {
                return Pair.of("Outputs differ: " + outputDiff,i);
            }
        }

        return null;  // No significant difference found
    }

    private static String compareWithEpsilon(Map<Integer, String> map1, Map<Integer, String> map2, double epsilon) {
        for (Integer key : map1.keySet()) {
            if (!map2.containsKey(key)) continue;  // Ignore keys not present in both maps

            String value1 = map1.get(key);
            String value2 = map2.get(key);

            //dup bug, ignore
            if(value1.contains("[]") || value2.contains("[]")) continue;
            try {
                JSONArray arr1 = new JSONArray(value1);
                JSONArray arr2 = new JSONArray(value2);

                String arrayDiff = compareArraysWithEpsilon(arr1, arr2, epsilon);
                if (arrayDiff != null) {
                    return "Key " + key + ": " + arrayDiff;
                }
            } catch (JSONException e) {
                // If not a JSON array, compare as individual values
                try {
                    double d1 = Double.parseDouble(value1);
                    double d2 = Double.parseDouble(value2);

                    if (Math.abs(d1 - d2) > epsilon) {
                        return String.format("Key %d: %f vs %f", key, d1, d2);
                    }
                } catch (NumberFormatException nfe) {
                    // If values are not numbers, compare them as strings
                    if (!value1.equals(value2)) {
                        return String.format("Key %d: %s vs %s", key, value1, value2);
                    }
                }
            }
        }

        return null;  // No significant difference found
    }

    private static String compareArraysWithEpsilon(JSONArray arr1, JSONArray arr2, double epsilon) throws JSONException {
        if (arr1.length() != arr2.length()) {
            return "Array lengths differ: " + arr1.length() + " vs " + arr2.length();
        }

        for (int i = 0; i < arr1.length(); i++) {
            Object val1 = arr1.get(i);
            Object val2 = arr2.get(i);

            if (val1 instanceof JSONArray && val2 instanceof JSONArray) {
                String nestedDiff = compareArraysWithEpsilon((JSONArray) val1, (JSONArray) val2, epsilon);
                if (nestedDiff != null) {
                    return "Nested array at index " + i + ": " + nestedDiff;
                }
            } else if (val1 instanceof Number && val2 instanceof Number) {
                double d1 = ((Number) val1).doubleValue();
                double d2 = ((Number) val2).doubleValue();
                if (Math.abs(d1 - d2) > epsilon) {
                    return String.format("Index %d: %f vs %f", i, d1, d2);
                }
            } else if (!val1.equals(val2)) {
                return String.format("Index %d: %s vs %s", i, val1, val2);
            }
        }

        return null;  // No significant difference found
    }
    private static Map<String, List<OpLogEvent>> loadAndNormalizeEvents(String jdbcUrl) throws SQLException {
        Map<String, List<OpLogEvent>> events = new LinkedHashMap<>();
        String query = "SELECT id, sourceCodeLine, opName, inputs, outputs, stackTrace FROM OpLogEvent ORDER BY id";

        try (Connection conn = DriverManager.getConnection(jdbcUrl, InterceptorEnvironment.USER, InterceptorEnvironment.PASSWORD);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(query)) {
            processResultSet(rs, events);
        }

        // Normalize and deduplicate events
        for (Map.Entry<String, List<OpLogEvent>> entry : events.entrySet()) {
            List<OpLogEvent> normalizedEvents = normalizeAndDeduplicateEvents(entry.getValue());
            entry.setValue(normalizedEvents);
        }

        return events;
    }

    private static List<OpLogEvent> normalizeAndDeduplicateEvents(List<OpLogEvent> events) {
        Set<OpLogEvent> normalizedSet = new LinkedHashSet<>();
        for (OpLogEvent event : events) {
            OpLogEvent normalizedEvent = OpLogEvent.builder()
                    .eventId(0L) // Set eventId to 0
                    .firstNonExecutionCodeLine(event.getFirstNonExecutionCodeLine())
                    .opName(event.getOpName())
                    .inputs(event.getInputs())
                    .outputs(event.getOutputs())
                    .stackTrace(event.getStackTrace())
                    .build();
            normalizedSet.add(normalizedEvent);
        }
        return new ArrayList<>(normalizedSet);
    }



    private static OpDifference compareInputs(Map<Integer, String> inputs1, Map<Integer, String> inputs2, double epsilon, OpLogEvent opLogEvent1, OpLogEvent opLogEvent2) {
        for (int j = 0; j < Math.min(inputs1.size(), inputs2.size()); j++) {
            JSONArray jsonArray1 = new JSONArray(inputs1.get(j));
            JSONArray jsonArray2 = new JSONArray(inputs2.get(j));
            JSONComparisonResult result = compareJSONArraysWithEpsilon(jsonArray1, jsonArray2, epsilon);
            if (!result.isSame()) {
                return OpDifference.builder()
                        .opLog1(opLogEvent1)
                        .opLog2(opLogEvent2)
                        .differenceType("inputs")
                        .differenceValue1(String.valueOf(result.getFirstValue()))
                        .differenceValue2(String.valueOf(result.getSecondValue()))
                        .opDifference(j)
                        .build();
            }
        }
        return null;
    }

    private static OpDifference compareOutputs(Map<Integer, String> outputs1, Map<Integer, String> outputs2, double epsilon, OpLogEvent opLogEvent1, OpLogEvent opLogEvent2) {
        for (int j = 0; j < Math.min(outputs1.size(), outputs2.size()); j++) {
            Object cast1 = parseOutput(outputs1.get(j));
            Object cast2 = parseOutput(outputs2.get(j));

            JSONArray casted1 = (JSONArray) cast1;
            JSONArray casted2 = (JSONArray) cast2;

            JSONComparisonResult result = compareJSONArraysWithEpsilon(casted1, casted2, epsilon);
            if (!result.isSame()) {
                return OpDifference.builder()
                        .opLog1(opLogEvent1)
                        .opLog2(opLogEvent2)
                        .differenceType("outputs")
                        .differenceValue1(String.valueOf(result.getFirstValue()))
                        .differenceValue2(String.valueOf(result.getSecondValue()))
                        .opDifference(result.getIndex())
                        .build();
            }
        }
        return null;
    }

    private static Object parseOutput(String output) {
        if (output.matches("-*\\d+\\.\\d+")) {
            return new JSONArray(new double[]{Double.parseDouble(output)});
        } else {
            return new JSONArray(output);
        }
    }

    private static JSONComparisonResult compareJSONArraysWithEpsilon(JSONArray arr1, JSONArray arr2, double epsilon) {
        if (arr1.length() != arr2.length()) {
            return JSONComparisonResult.builder().same(false).index(-1).build();
        }

        for (int i = 0; i < arr1.length(); i++) {
            if (arr1.get(i) instanceof JSONArray && arr2.get(i) instanceof JSONArray) {
                JSONComparisonResult result = compareJSONArraysWithEpsilon(arr1.getJSONArray(i), arr2.getJSONArray(i), epsilon);
                if (!result.isSame()) {
                    return result;
                }
            } else {
                double val1 = arr1.getDouble(i);
                double val2 = arr2.getDouble(i);
                if (Math.abs(val1 - val2) > epsilon) {
                    return JSONComparisonResult.builder()
                            .same(false)
                            .index(i)
                            .firstValue(val1)
                            .secondValue(val2)
                            .build();
                }
            }
        }

        return JSONComparisonResult.builder().same(true).build();
    }

    private static boolean isValidDifference(OpDifference diff) {
        if (diff == null) {
            return false;
        }
        if (diff.getOpLog1() == null || diff.getOpLog2() == null) {
            return false;
        }
        if (diff.getOpDifference() == -1) {
            return false;
        }
        return true;
    }

    private static boolean isSignificantDifference(OpDifference diff, double epsilon) {
        if (!isValidDifference(diff)) {
            return false;
        }
        if (diff.getDifferenceType().equals("size") || diff.getDifferenceType().equals("missing_line")) {
            return true;
        }
        try {
            double value1 = Double.parseDouble(diff.getDifferenceValue1());
            double value2 = Double.parseDouble(diff.getDifferenceValue2());
            return Math.abs(value1 - value2) > epsilon;
        } catch (NumberFormatException e) {
            // If we can't parse the values as doubles, consider it significant
            return true;
        }
    }

    private static OpDifference updateEarliestDifference(OpDifference currentEarliest, OpDifference newDifference) {
        if (currentEarliest == null) {
            return newDifference;
        }

        long currentEarliestTime = getEarliestTime(currentEarliest);
        long newDifferenceTime = getEarliestTime(newDifference);

        return newDifferenceTime < currentEarliestTime ? newDifference : currentEarliest;
    }

    private static long getEarliestTime(OpDifference diff) {
        return Math.min(
                diff.getOpLog1() != null ? diff.getOpLog1().getEventId() : Long.MAX_VALUE,
                diff.getOpLog2() != null ? diff.getOpLog2().getEventId() : Long.MAX_VALUE
        );
    }

    private static List<Map.Entry<String, OpDifference>> sortDifferences(Map<String, OpDifference> differences) {
        List<Map.Entry<String, OpDifference>> sortedDifferences = new ArrayList<>(differences.entrySet());
        sortedDifferences.sort((e1, e2) -> {
            long time1 = getEarliestTime(e1.getValue());
            long time2 = getEarliestTime(e2.getValue());
            return Long.compare(time1, time2);
        });
        return sortedDifferences;
    }

    private static void printDifferences(Map<String, OpDifference> differences) {
        for (Map.Entry<String, OpDifference> entry : differences.entrySet()) {
            System.out.println("Line: " + entry.getKey());
            OpDifference diff = entry.getValue();
            System.out.println("  Earliest Difference Type: " + diff.getDifferenceType());
            System.out.println("  Earliest Event ID: " + getEarliestTime(diff));
            System.out.println();
        }
    }

    private static void printSortedDifferences(List<Map.Entry<String, OpDifference>> sortedDifferences) {
        for (Map.Entry<String, OpDifference> entry : sortedDifferences) {
            String line = entry.getKey();
            OpDifference diff = entry.getValue();
            long earliestTime = getEarliestTime(diff);
            System.out.println("Line: " + line);
            System.out.println("  Earliest Difference Type: " + diff.getDifferenceType());
            System.out.println("  Earliest Event ID: " + earliestTime);
            System.out.println("  Op Name: " + (diff.getOpLog1() != null ? diff.getOpLog1().getOpName() : diff.getOpLog2().getOpName()));
            System.out.println("  Difference Value 1: " + diff.getDifferenceValue1());
            System.out.println("  Difference Value 2: " + diff.getDifferenceValue2());
            System.out.println();
        }
    }
}
