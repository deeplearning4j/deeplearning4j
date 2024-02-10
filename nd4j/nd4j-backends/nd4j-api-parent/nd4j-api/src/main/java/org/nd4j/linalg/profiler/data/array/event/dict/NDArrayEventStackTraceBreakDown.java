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
package org.nd4j.linalg.profiler.data.array.event.dict;

import org.nd4j.common.primitives.Triple;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceElementCache;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceLookupKey;
import org.nd4j.shade.guava.collect.Table;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class NDArrayEventStackTraceBreakDown extends ConcurrentHashMap<StackTraceElement, Table<StackTraceElement, StackTraceElement, List<NDArrayEvent>>> {





    public String displayStackTraceHierarchy() {
        Iterator<Triple<StackTraceElement, StackTraceElement, StackTraceElement>> tripleIterator = enumerateEntries();
        StringBuilder ret = new StringBuilder();
        while(tripleIterator.hasNext()) {
            Triple<StackTraceElement,StackTraceElement,StackTraceElement> next = tripleIterator.next();
            StackTraceElement row = next.getFirst();
            StackTraceElement column = next.getSecond();
            StackTraceElement value = next.getThird();
            ret.append("\n" + row + "\n");
            ret.append("\t" + column + "\n");
            ret.append("\t\t" + value + "\n\n\n");
        }

        return ret.toString();
    }



    public List<NDArrayEvent> getEvents(StackTraceLookupKey row,
                                        StackTraceLookupKey column,
                                        StackTraceLookupKey value) {
        return getEvents(StackTraceElementCache.lookup(row),StackTraceElementCache.lookup(column),StackTraceElementCache.lookup(value));
    }

    public List<NDArrayEvent> getEvents(StackTraceElement tableKey,
                                        StackTraceElement row,
                                        StackTraceElement column) {
        if(!this.containsKey(tableKey)) {
            return new ArrayList<>();
        }

        Table<StackTraceElement, StackTraceElement, List<NDArrayEvent>> table = get(tableKey);
        if(table == null || !table.containsRow(row)) {
            return new ArrayList<>();
        }

        List<NDArrayEvent> ret = table.get(row,column);
        if(ret == null) {
            return new ArrayList<>();
        }

        return ret;

    }

    public Iterator<Triple<StackTraceElement,StackTraceElement,StackTraceElement>> enumerateEntries() {
        List<Triple<StackTraceElement,StackTraceElement,StackTraceElement>> ret = new ArrayList<>();
        for(StackTraceElement tableKey : keySet()) {
            Table<StackTraceElement, StackTraceElement, List<NDArrayEvent>> table = get(tableKey);
            for(StackTraceElement row : table.rowKeySet()) {
                for(StackTraceElement column : table.columnKeySet()) {
                    ret.add(new Triple<>(tableKey,row,column));
                }
            }
        }
        return ret.iterator();
    }





    public  BreakDownComparison compareBreakDown(BreakdownArgs breakdownArgs) {
        StackTraceElement targetTable = StackTraceElementCache.lookup(breakdownArgs.getPointOfOrigin());
        StackTraceElement compTable = StackTraceElementCache.lookup(breakdownArgs.getCompPointOfOrigin());
        StackTraceElement targetRow = StackTraceElementCache.lookup(breakdownArgs.getCommonPointOfInvocation());
        StackTraceElement targetColumn = StackTraceElementCache.lookup(breakdownArgs.getCommonParentOfInvocation());

        if(targetTable == null || compTable == null || targetRow == null || targetColumn == null) {
            return BreakDownComparison.empty();
        }

        StringBuilder stringBuilder = new StringBuilder();
        if(!containsKey(targetTable) || !containsKey(compTable)) {
            return  BreakDownComparison.empty();
        }

        Table<StackTraceElement, StackTraceElement, List<NDArrayEvent>> firstTable = get(targetTable);
        Map<StackTraceElement, List<NDArrayEvent>> targetTableRow = firstTable.row(targetRow);
        Table<StackTraceElement, StackTraceElement, List<NDArrayEvent>> secondTable = get(compTable);
        Map<StackTraceElement, List<NDArrayEvent>> compTableRow = secondTable.row(targetRow);
        if(targetTableRow == null || compTableRow == null) {
            return  BreakDownComparison.empty();
        }



        if(!targetTableRow.containsKey(targetColumn) || !compTableRow.containsKey(targetColumn)) {
            StringBuilder stringBuilder1 = new StringBuilder();
            stringBuilder1.append("First table: " + targetTableRow + "\n");
            stringBuilder1.append("Second table: " + compTableRow + "\n");
            stringBuilder.append("Unable to compare data. The following table results were found:\n");
            return  BreakDownComparison.empty();
        }



        List<NDArrayEvent> targetEvents = targetTableRow.get(targetColumn);
        List<NDArrayEvent> compEvents = compTableRow.get(targetColumn);
        targetEvents = breakdownArgs.filterIfNeeded(targetEvents);
        compEvents = breakdownArgs.filterIfNeeded(compEvents);

        return BreakDownComparison.builder()
                .first(targetEvents)
                .second(compEvents)
                .build();
    }

}
