package org.nd4j.autodiff.samediff;/*
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


import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Methods for retrieving and analyzing loop termination events
 */
@Slf4j
public class LoopTerminationEventUtils {

    /**
     * Get the latest termination event for a specific frame
     *
     * @param frameName The name of the loop frame
     * @param terminationHistory Map of frame names to their termination events
     * @return The most recent termination event for the frame, or null if none exists
     */
    public static LoopTerminationEvent getLatestTerminationEvent(String frameName,
                                                                 Map<String, List<LoopTerminationEvent>> terminationHistory) {
        List<LoopTerminationEvent> events = terminationHistory.get(frameName);
        if (events != null && !events.isEmpty()) {
            return events.get(events.size() - 1);
        }
        return null;
    }

    /**
     * Get the latest termination event for a specific frame, with additional filtering
     *
     * @param frameName The name of the loop frame
     * @param terminationHistory Map of frame names to their termination events
     * @param terminationType Filter by specific termination type (null for any type)
     * @return The most recent termination event matching criteria, or null if none exists
     */
    public static LoopTerminationEvent getLatestTerminationEvent(String frameName,
                                                                 Map<String, List<LoopTerminationEvent>> terminationHistory,
                                                                 TerminationType terminationType) {
        List<LoopTerminationEvent> events = terminationHistory.get(frameName);
        if (events != null && !events.isEmpty()) {
            // Filter by termination type if specified
            if (terminationType != null) {
                List<LoopTerminationEvent> filteredEvents = events.stream()
                        .filter(event -> event.getTerminationType() == terminationType)
                        .collect(Collectors.toList());

                if (!filteredEvents.isEmpty()) {
                    return filteredEvents.get(filteredEvents.size() - 1);
                }
            } else {
                return events.get(events.size() - 1);
            }
        }
        return null;
    }

    /**
     * Get the latest termination event across all frames
     *
     * @param terminationHistory Map of frame names to their termination events
     * @return The most recent termination event across all frames, or null if none exists
     */
    public static LoopTerminationEvent getLatestTerminationEventGlobal(Map<String, List<LoopTerminationEvent>> terminationHistory) {
        LoopTerminationEvent latestEvent = null;
        long latestTimestamp = 0;

        for (Map.Entry<String, List<LoopTerminationEvent>> entry : terminationHistory.entrySet()) {
            List<LoopTerminationEvent> events = entry.getValue();
            if (events != null && !events.isEmpty()) {
                LoopTerminationEvent lastEvent = events.get(events.size() - 1);
                if (lastEvent.getTimestamp() > latestTimestamp) {
                    latestTimestamp = lastEvent.getTimestamp();
                    latestEvent = lastEvent;
                }
            }
        }

        return latestEvent;
    }

    /**
     * Get the latest early termination event for a specific frame
     *
     * @param frameName The name of the loop frame
     * @param terminationHistory Map of frame names to their termination events
     * @return The most recent early termination event, or null if none exists
     */
    public static LoopTerminationEvent getLatestEarlyTerminationEvent(String frameName,
                                                                      Map<String, List<LoopTerminationEvent>> terminationHistory) {
        List<LoopTerminationEvent> events = terminationHistory.get(frameName);
        if (events != null && !events.isEmpty()) {
            // Find the latest early termination event
            for (int i = events.size() - 1; i >= 0; i--) {
                LoopTerminationEvent event = events.get(i);
                if (event.isWasEarlyTermination()) {
                    return event;
                }
            }
        }
        return null;
    }

    /**
     * Get the latest termination event by timestamp for a specific frame
     *
     * @param frameName The name of the loop frame
     * @param terminationHistory Map of frame names to their termination events
     * @return The termination event with the highest timestamp, or null if none exists
     */
    public static LoopTerminationEvent getLatestTerminationEventByTimestamp(String frameName,
                                                                            Map<String, List<LoopTerminationEvent>> terminationHistory) {
        List<LoopTerminationEvent> events = terminationHistory.get(frameName);
        if (events != null && !events.isEmpty()) {
            return events.stream()
                    .max(Comparator.comparingLong(LoopTerminationEvent::getTimestamp))
                    .orElse(null);
        }
        return null;
    }

    /**
     * Get the latest termination event with detailed context information
     *
     * @param frameName The name of the loop frame
     * @param terminationHistory Map of frame names to their termination events
     * @return Enhanced termination event information, or null if none exists
     */
    public static EnhancedTerminationEvent getLatestTerminationEventWithContext(String frameName,
                                                                                Map<String, List<LoopTerminationEvent>> terminationHistory) {
        List<LoopTerminationEvent> events = terminationHistory.get(frameName);
        if (events != null && !events.isEmpty()) {
            LoopTerminationEvent latestEvent = events.get(events.size() - 1);

            // Create enhanced event with additional context
            EnhancedTerminationEvent enhancedEvent = new EnhancedTerminationEvent(latestEvent);

            // Add context about previous events
            if (events.size() > 1) {
                enhancedEvent.setPreviousEvent(events.get(events.size() - 2));
                enhancedEvent.setEventSequenceNumber(events.size());
            }

            // Add timing analysis
            if (events.size() > 1) {
                long timeBetweenEvents = latestEvent.getTimestamp() - events.get(0).getTimestamp();
                enhancedEvent.setTotalExecutionTime(timeBetweenEvents);
            }

            // Add pattern analysis
            enhancedEvent.setTerminationPattern(analyzeTerminationPattern(events));

            return enhancedEvent;
        }
        return null;
    }

    /**
     * Get all termination events for a frame, sorted by timestamp
     *
     * @param frameName The name of the loop frame
     * @param terminationHistory Map of frame names to their termination events
     * @return List of termination events sorted by timestamp (oldest first)
     */
    public static List<LoopTerminationEvent> getAllTerminationEventsSorted(String frameName,
                                                                           Map<String, List<LoopTerminationEvent>> terminationHistory) {
        List<LoopTerminationEvent> events = terminationHistory.get(frameName);
        if (events != null && !events.isEmpty()) {
            return events.stream()
                    .sorted(Comparator.comparingLong(LoopTerminationEvent::getTimestamp))
                    .collect(Collectors.toList());
        }
        return List.of();
    }

    /**
     * Get termination events within a specific time range
     *
     * @param frameName The name of the loop frame
     * @param terminationHistory Map of frame names to their termination events
     * @param startTime Start of time range (inclusive)
     * @param endTime End of time range (inclusive)
     * @return List of termination events within the time range
     */
    public static List<LoopTerminationEvent> getTerminationEventsInTimeRange(String frameName,
                                                                             Map<String, List<LoopTerminationEvent>> terminationHistory,
                                                                             long startTime, long endTime) {
        List<LoopTerminationEvent> events = terminationHistory.get(frameName);
        if (events != null && !events.isEmpty()) {
            return events.stream()
                    .filter(event -> event.getTimestamp() >= startTime && event.getTimestamp() <= endTime)
                    .collect(Collectors.toList());
        }
        return List.of();
    }

    /**
     * Get termination events by type
     *
     * @param frameName The name of the loop frame
     * @param terminationHistory Map of frame names to their termination events
     * @param terminationType The type of termination to filter by
     * @return List of termination events of the specified type
     */
    public static List<LoopTerminationEvent> getTerminationEventsByType(String frameName,
                                                                        Map<String, List<LoopTerminationEvent>> terminationHistory,
                                                                        TerminationType terminationType) {
        List<LoopTerminationEvent> events = terminationHistory.get(frameName);
        if (events != null && !events.isEmpty()) {
            return events.stream()
                    .filter(event -> event.getTerminationType() == terminationType)
                    .collect(Collectors.toList());
        }
        return List.of();
    }

    /**
     * Get the most recent termination event that matches specific criteria
     *
     * @param frameName The name of the loop frame
     * @param terminationHistory Map of frame names to their termination events
     * @param criteria Predicate to filter events
     * @return The most recent event matching criteria, or null if none exists
     */
    public static LoopTerminationEvent getLatestTerminationEventMatching(String frameName,
                                                                         Map<String, List<LoopTerminationEvent>> terminationHistory,
                                                                         java.util.function.Predicate<LoopTerminationEvent> criteria) {
        List<LoopTerminationEvent> events = terminationHistory.get(frameName);
        if (events != null && !events.isEmpty()) {
            // Search from most recent to oldest
            for (int i = events.size() - 1; i >= 0; i--) {
                LoopTerminationEvent event = events.get(i);
                if (criteria.test(event)) {
                    return event;
                }
            }
        }
        return null;
    }

    /**
     * Check if a frame has any termination events
     *
     * @param frameName The name of the loop frame
     * @param terminationHistory Map of frame names to their termination events
     * @return true if the frame has termination events, false otherwise
     */
    public static boolean hasTerminationEvents(String frameName,
                                               Map<String, List<LoopTerminationEvent>> terminationHistory) {
        List<LoopTerminationEvent> events = terminationHistory.get(frameName);
        return events != null && !events.isEmpty();
    }

    /**
     * Get the count of termination events for a frame
     *
     * @param frameName The name of the loop frame
     * @param terminationHistory Map of frame names to their termination events
     * @return Number of termination events for the frame
     */
    public static int getTerminationEventCount(String frameName,
                                               Map<String, List<LoopTerminationEvent>> terminationHistory) {
        List<LoopTerminationEvent> events = terminationHistory.get(frameName);
        return events != null ? events.size() : 0;
    }

    /**
     * Get summary statistics for termination events
     *
     * @param frameName The name of the loop frame
     * @param terminationHistory Map of frame names to their termination events
     * @return TerminationEventSummary with statistics
     */
    public static TerminationEventSummary getTerminationEventSummary(String frameName,
                                                                     Map<String, List<LoopTerminationEvent>> terminationHistory) {
        List<LoopTerminationEvent> events = terminationHistory.get(frameName);
        if (events == null || events.isEmpty()) {
            return new TerminationEventSummary(frameName, 0, 0, 0, null, null);
        }

        long totalEvents = events.size();
        long earlyTerminations = events.stream()
                .mapToLong(event -> event.isWasEarlyTermination() ? 1 : 0)
                .sum();

        long errorTerminations = events.stream()
                .mapToLong(event -> event.getTerminationType() == TerminationType.ERROR_TERMINATION ? 1 : 0)
                .sum();

        LoopTerminationEvent firstEvent = events.get(0);
        LoopTerminationEvent lastEvent = events.get(events.size() - 1);

        return new TerminationEventSummary(frameName, totalEvents, earlyTerminations,
                errorTerminations, firstEvent, lastEvent);
    }

    // Helper method to analyze termination patterns
    private static String analyzeTerminationPattern(List<LoopTerminationEvent> events) {
        if (events.size() < 2) {
            return "SINGLE_EVENT";
        }

        // Check for consistent termination types
        TerminationType lastType = events.get(events.size() - 1).getTerminationType();
        boolean allSameType = events.stream()
                .allMatch(event -> event.getTerminationType() == lastType);

        if (allSameType) {
            return "CONSISTENT_" + lastType.name();
        }

        // Check for alternating patterns
        boolean alternating = true;
        for (int i = 1; i < events.size(); i++) {
            if (events.get(i).getTerminationType() == events.get(i - 1).getTerminationType()) {
                alternating = false;
                break;
            }
        }

        if (alternating) {
            return "ALTERNATING_PATTERN";
        }

        return "MIXED_PATTERN";
    }

    /**
     * Enhanced termination event with additional context
     */
    public static class EnhancedTerminationEvent {
        private final LoopTerminationEvent baseEvent;
        private LoopTerminationEvent previousEvent;
        private int eventSequenceNumber;
        private long totalExecutionTime;
        private String terminationPattern;

        public EnhancedTerminationEvent(LoopTerminationEvent baseEvent) {
            this.baseEvent = baseEvent;
        }

        // Getters and setters
        public LoopTerminationEvent getBaseEvent() { return baseEvent; }
        public LoopTerminationEvent getPreviousEvent() { return previousEvent; }
        public void setPreviousEvent(LoopTerminationEvent previousEvent) { this.previousEvent = previousEvent; }
        public int getEventSequenceNumber() { return eventSequenceNumber; }
        public void setEventSequenceNumber(int eventSequenceNumber) { this.eventSequenceNumber = eventSequenceNumber; }
        public long getTotalExecutionTime() { return totalExecutionTime; }
        public void setTotalExecutionTime(long totalExecutionTime) { this.totalExecutionTime = totalExecutionTime; }
        public String getTerminationPattern() { return terminationPattern; }
        public void setTerminationPattern(String terminationPattern) { this.terminationPattern = terminationPattern; }

        public String getEnhancedSummary() {
            StringBuilder summary = new StringBuilder();
            summary.append("Event #").append(eventSequenceNumber);
            summary.append(" - ").append(baseEvent.getTerminationType());
            summary.append(" at iteration ").append(baseEvent.getIteration());

            if (previousEvent != null) {
                long timeBetween = baseEvent.getTimestamp() - previousEvent.getTimestamp();
                summary.append(" (").append(timeBetween).append("ms after previous)");
            }

            if (terminationPattern != null) {
                summary.append(" [").append(terminationPattern).append("]");
            }

            return summary.toString();
        }
    }

    /**
     * Summary statistics for termination events
     */
    public static class TerminationEventSummary {
        private final String frameName;
        private final long totalEvents;
        private final long earlyTerminations;
        private final long errorTerminations;
        private final LoopTerminationEvent firstEvent;
        private final LoopTerminationEvent lastEvent;

        public TerminationEventSummary(String frameName, long totalEvents, long earlyTerminations,
                                       long errorTerminations, LoopTerminationEvent firstEvent,
                                       LoopTerminationEvent lastEvent) {
            this.frameName = frameName;
            this.totalEvents = totalEvents;
            this.earlyTerminations = earlyTerminations;
            this.errorTerminations = errorTerminations;
            this.firstEvent = firstEvent;
            this.lastEvent = lastEvent;
        }

        // Getters
        public String getFrameName() { return frameName; }
        public long getTotalEvents() { return totalEvents; }
        public long getEarlyTerminations() { return earlyTerminations; }
        public long getErrorTerminations() { return errorTerminations; }
        public LoopTerminationEvent getFirstEvent() { return firstEvent; }
        public LoopTerminationEvent getLastEvent() { return lastEvent; }

        public double getEarlyTerminationRate() {
            return totalEvents > 0 ? (double) earlyTerminations / totalEvents : 0.0;
        }

        public double getErrorTerminationRate() {
            return totalEvents > 0 ? (double) errorTerminations / totalEvents : 0.0;
        }

        public long getTotalExecutionTime() {
            if (firstEvent != null && lastEvent != null) {
                return lastEvent.getTimestamp() - firstEvent.getTimestamp();
            }
            return 0;
        }

        @Override
        public String toString() {
            return String.format("TerminationEventSummary[frame=%s, total=%d, early=%d, errors=%d, earlyRate=%.2f%%]",
                    frameName, totalEvents, earlyTerminations, errorTerminations,
                    getEarlyTerminationRate() * 100);
        }
    }



    /**
     * Get all early termination events across all frames
     */
    public static List<LoopTerminationEvent> getAllEarlyTerminationEvents(
            Map<String, List<LoopTerminationEvent>> terminationHistory) {
        return terminationHistory.values().stream()
                .flatMap(List::stream)
                .filter(LoopTerminationEvent::isWasEarlyTermination)
                .collect(Collectors.toList());
    }

    /**
     * Find termination events that occurred within a time window
     */
    public static List<LoopTerminationEvent> getTerminationEventsInTimeWindow(
            Map<String, List<LoopTerminationEvent>> terminationHistory,
            long startTime, long endTime) {
        return terminationHistory.values().stream()
                .flatMap(List::stream)
                .filter(event -> event.getTimestamp() >= startTime && event.getTimestamp() <= endTime)
                .collect(Collectors.toList());
    }

    /**
     * Group termination events by their termination type
     */
    public static Map<TerminationType, List<LoopTerminationEvent>> groupEventsByType(
            Map<String, List<LoopTerminationEvent>> terminationHistory) {
        return terminationHistory.values().stream()
                .flatMap(List::stream)
                .collect(Collectors.groupingBy(LoopTerminationEvent::getTerminationType));
    }

    /**
     * Calculate termination statistics
     */
    public static Map<String, Object> calculateTerminationStatistics(
            Map<String, List<LoopTerminationEvent>> terminationHistory) {
        Map<String, Object> stats = new HashMap<>();

        List<LoopTerminationEvent> allEvents = terminationHistory.values().stream()
                .flatMap(List::stream)
                .collect(Collectors.toList());

        stats.put("totalEvents", allEvents.size());
        stats.put("uniqueFrames", terminationHistory.keySet().size());

        // Count by type
        Map<TerminationType, Long> typeCounts = allEvents.stream()
                .collect(Collectors.groupingBy(LoopTerminationEvent::getTerminationType, Collectors.counting()));
        stats.put("eventsByType", typeCounts);

        // Early termination rate
        long earlyTerminations = allEvents.stream()
                .filter(LoopTerminationEvent::isWasEarlyTermination)
                .count();
        double earlyTerminationRate = allEvents.size() > 0 ? (double) earlyTerminations / allEvents.size() : 0.0;
        stats.put("earlyTerminationRate", earlyTerminationRate);

        // Average iteration at termination
        double avgIteration = allEvents.stream()
                .mapToInt(LoopTerminationEvent::getIteration)
                .average()
                .orElse(0.0);
        stats.put("averageTerminationIteration", avgIteration);

        return stats;
    }
}