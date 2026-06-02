/*
 * *****************************************************************************
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 * ****************************************************************************
 */

package org.nd4j.common.tests.results;

import java.nio.file.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Queries persisted test results to find regressions, history, and summaries.
 * <p>
 * The data lives in {@code ~/.dl4j/test-results/} as JSONL files. Each line is
 * a self-contained JSON object. This class does simple string parsing — no
 * JSON library dependency.
 *
 * <h3>CLI usage (from platform-tests):</h3>
 * <pre>
 * # Show today's failures
 * mvn exec:java -Dexec.mainClass=org.nd4j.common.tests.results.TestResultQuery -Dexec.args="failures"
 *
 * # Show regressions between two dates
 * mvn exec:java -Dexec.mainClass=org.nd4j.common.tests.results.TestResultQuery -Dexec.args="regressions 2026-05-28 2026-05-30"
 *
 * # Show history of a specific test
 * mvn exec:java -Dexec.mainClass=org.nd4j.common.tests.results.TestResultQuery -Dexec.args="history SomeDspTest"
 *
 * # Show summary for a date
 * mvn exec:java -Dexec.mainClass=org.nd4j.common.tests.results.TestResultQuery -Dexec.args="summary 2026-05-30"
 *
 * # Show all milestones
 * mvn exec:java -Dexec.mainClass=org.nd4j.common.tests.results.TestResultQuery -Dexec.args="milestones"
 * </pre>
 */
public class TestResultQuery {

    private static final DateTimeFormatter DATE_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    private TestResultQuery() {}

    // ─── Structured result record ───────────────────────────────

    /**
     * Lightweight parsed representation of one test result line.
     */
    public static class TestResult {
        public final String timestamp;
        public final String commit;
        public final String branch;
        public final String backend;
        public final String testClass;
        public final String testMethod;
        public final String displayName;
        public final String status;
        public final long durationMs;
        public final String error;

        TestResult(String line) {
            this.timestamp = extractString(line, "ts");
            this.commit = extractString(line, "commit");
            this.branch = extractString(line, "branch");
            this.backend = extractString(line, "backend");
            this.testClass = extractString(line, "testClass");
            this.testMethod = extractString(line, "testMethod");
            this.displayName = extractString(line, "displayName");
            this.status = extractString(line, "status");
            this.durationMs = extractLong(line, "durationMs");
            this.error = extractString(line, "error");
        }

        /** Unique test identity: class#method */
        public String testId() {
            return testMethod != null ? testClass + "#" + testMethod : testClass;
        }

        @Override
        public String toString() {
            String id = testId();
            // Show just the simple class name for readability
            int lastDot = id.lastIndexOf('.');
            String shortId = lastDot >= 0 ? id.substring(lastDot + 1) : id;
            return String.format("%-6s %6dms  %-50s  [%s %s %s]",
                    status, durationMs, shortId, backend, commit, timestamp);
        }
    }

    // ─── Query methods ──────────────────────────────────────────

    /**
     * Returns all test results for a given date, parsed into TestResult objects.
     */
    public static List<TestResult> getResults(LocalDate date) {
        return TestResultStore.readResults(date).stream()
                .filter(line -> line.contains("\"type\":\"test\""))
                .map(TestResult::new)
                .collect(Collectors.toList());
    }

    /**
     * Returns all results across all available dates.
     */
    public static List<TestResult> getAllResults() {
        List<TestResult> all = new ArrayList<>();
        for (Path file : TestResultStore.listResultFiles()) {
            String name = file.getFileName().toString();
            // Parse date from filename: results-2026-05-30.jsonl
            String dateStr = name.replace("results-", "").replace(".jsonl", "");
            try {
                LocalDate date = LocalDate.parse(dateStr, DATE_FMT);
                all.addAll(getResults(date));
            } catch (Exception e) {
                // Skip malformed filenames
            }
        }
        return all;
    }

    /**
     * Returns failures from a given date.
     */
    public static List<TestResult> getFailures(LocalDate date) {
        return getResults(date).stream()
                .filter(r -> "FAIL".equals(r.status))
                .collect(Collectors.toList());
    }

    /**
     * Finds regressions: tests that passed on {@code olderDate} but failed on {@code newerDate}.
     * Matches by testId (class#method) and backend.
     */
    public static List<TestResult> getRegressions(LocalDate olderDate, LocalDate newerDate) {
        Map<String, TestResult> olderByKey = new HashMap<>();
        for (TestResult r : getResults(olderDate)) {
            if ("PASS".equals(r.status)) {
                olderByKey.put(r.testId() + "@" + r.backend, r);
            }
        }

        return getResults(newerDate).stream()
                .filter(r -> "FAIL".equals(r.status))
                .filter(r -> olderByKey.containsKey(r.testId() + "@" + r.backend))
                .collect(Collectors.toList());
    }

    /**
     * Returns the history of a specific test (by class name substring match) across all dates.
     */
    public static List<TestResult> getHistory(String classNameSubstring) {
        return getAllResults().stream()
                .filter(r -> r.testClass.contains(classNameSubstring)
                        || (r.testMethod != null && r.testMethod.contains(classNameSubstring)))
                .collect(Collectors.toList());
    }

    /**
     * Returns tests that have both passed and failed across all recorded history.
     * These are likely flaky.
     */
    public static Set<String> getFlakyTests() {
        Map<String, Set<String>> statusesByTest = new HashMap<>();
        for (TestResult r : getAllResults()) {
            statusesByTest.computeIfAbsent(r.testId() + "@" + r.backend, k -> new HashSet<>())
                    .add(r.status);
        }
        return statusesByTest.entrySet().stream()
                .filter(e -> e.getValue().contains("PASS") && e.getValue().contains("FAIL"))
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());
    }

    /**
     * Generates a summary for a given date: counts by status and backend.
     */
    public static String summary(LocalDate date) {
        List<TestResult> results = getResults(date);
        if (results.isEmpty()) {
            return "No results for " + date.format(DATE_FMT);
        }

        Map<String, Map<String, Integer>> countsByBackend = new LinkedHashMap<>();
        for (TestResult r : results) {
            countsByBackend.computeIfAbsent(r.backend, k -> new LinkedHashMap<>())
                    .merge(r.status, 1, Integer::sum);
        }

        StringBuilder sb = new StringBuilder();
        sb.append("=== Test Results for ").append(date.format(DATE_FMT)).append(" ===\n");
        sb.append("Total: ").append(results.size()).append(" results\n\n");

        for (Map.Entry<String, Map<String, Integer>> entry : countsByBackend.entrySet()) {
            sb.append("Backend: ").append(entry.getKey()).append("\n");
            for (Map.Entry<String, Integer> status : entry.getValue().entrySet()) {
                sb.append(String.format("  %-6s: %d\n", status.getKey(), status.getValue()));
            }
            sb.append("\n");
        }

        // List commits seen
        Set<String> commits = results.stream().map(r -> r.commit).collect(Collectors.toCollection(LinkedHashSet::new));
        sb.append("Commits: ").append(String.join(", ", commits)).append("\n");

        return sb.toString();
    }

    // ─── JSON parsing helpers (no library dependency) ───────────

    static String extractString(String json, String key) {
        String pattern = "\"" + key + "\":\"";
        int start = json.indexOf(pattern);
        if (start < 0) return null;
        start += pattern.length();
        int end = json.indexOf('"', start);
        if (end < 0) return null;
        return json.substring(start, end)
                .replace("\\n", "\n")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\");
    }

    static long extractLong(String json, String key) {
        String pattern = "\"" + key + "\":";
        int start = json.indexOf(pattern);
        if (start < 0) return 0;
        start += pattern.length();
        int end = start;
        while (end < json.length() && (Character.isDigit(json.charAt(end)) || json.charAt(end) == '-')) {
            end++;
        }
        try {
            return Long.parseLong(json.substring(start, end));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    // ─── CLI ────────────────────────────────────────────────────

    public static void main(String[] args) {
        if (args.length == 0) {
            printUsage();
            return;
        }

        String cmd = args[0];
        switch (cmd) {
            case "failures": {
                LocalDate date = args.length > 1 ? LocalDate.parse(args[1]) : LocalDate.now();
                List<TestResult> failures = getFailures(date);
                if (failures.isEmpty()) {
                    System.out.println("No failures for " + date);
                } else {
                    System.out.println("=== Failures for " + date + " (" + failures.size() + ") ===");
                    failures.forEach(System.out::println);
                }
                break;
            }
            case "regressions": {
                if (args.length < 3) {
                    System.out.println("Usage: regressions <older-date> <newer-date>");
                    return;
                }
                LocalDate older = LocalDate.parse(args[1]);
                LocalDate newer = LocalDate.parse(args[2]);
                List<TestResult> regressions = getRegressions(older, newer);
                if (regressions.isEmpty()) {
                    System.out.println("No regressions between " + older + " and " + newer);
                } else {
                    System.out.println("=== Regressions: " + older + " -> " + newer
                            + " (" + regressions.size() + ") ===");
                    regressions.forEach(System.out::println);
                }
                break;
            }
            case "history": {
                if (args.length < 2) {
                    System.out.println("Usage: history <class-name-substring>");
                    return;
                }
                List<TestResult> history = getHistory(args[1]);
                if (history.isEmpty()) {
                    System.out.println("No results matching '" + args[1] + "'");
                } else {
                    System.out.println("=== History for '" + args[1] + "' (" + history.size() + ") ===");
                    history.forEach(System.out::println);
                }
                break;
            }
            case "summary": {
                LocalDate date = args.length > 1 ? LocalDate.parse(args[1]) : LocalDate.now();
                System.out.println(summary(date));
                break;
            }
            case "milestones": {
                TestMilestone.printAll();
                break;
            }
            case "flaky": {
                Set<String> flaky = getFlakyTests();
                if (flaky.isEmpty()) {
                    System.out.println("No flaky tests detected (need multiple runs to detect)");
                } else {
                    System.out.println("=== Flaky Tests (" + flaky.size() + ") ===");
                    flaky.forEach(System.out::println);
                }
                break;
            }
            case "files": {
                List<Path> files = TestResultStore.listResultFiles();
                System.out.println("=== Result files (" + files.size() + ") in "
                        + TestResultStore.getResultsDir() + " ===");
                files.forEach(System.out::println);
                break;
            }
            default:
                printUsage();
        }
    }

    private static void printUsage() {
        System.out.println("Usage: TestResultQuery <command> [args]");
        System.out.println();
        System.out.println("Commands:");
        System.out.println("  failures [date]                  Show failures (default: today)");
        System.out.println("  regressions <older> <newer>      Tests that passed then failed");
        System.out.println("  history <class-substring>        All results for matching tests");
        System.out.println("  summary [date]                   Pass/fail/skip counts by backend");
        System.out.println("  milestones                       Show recorded milestones");
        System.out.println("  flaky                            Tests that both pass and fail");
        System.out.println("  files                            List result files");
        System.out.println();
        System.out.println("Dates: yyyy-MM-dd format (e.g. 2026-05-30)");
        System.out.println("Data dir: " + TestResultStore.getResultsDir());
    }
}
