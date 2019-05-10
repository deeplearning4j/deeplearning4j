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

package org.datavec.hadoop.records.reader.mapfile.index;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapFile;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ReflectionUtils;
import org.nd4j.linalg.primitives.Pair;
import org.datavec.hadoop.records.reader.mapfile.IndexToKey;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * A default implementation of {@link IndexToKey} that assumes (strictly requires) keys that are
 * {@link LongWritable} values, where all values are both unique and contiguous (0 to numRecords()-1)<br>
 * This allows for easy inference of the number of records, and identify mapping between indexes and keys.
 *
 * @author Alex Black
 */
public class LongIndexToKey implements IndexToKey {

    private List<Pair<Long, Long>> readerIndices;

    @Override
    public List<Pair<Long, Long>> initialize(MapFile.Reader[] readers, Class<? extends Writable> valueClass)
                    throws IOException {

        List<Pair<Long, Long>> l = new ArrayList<>(readers.length);
        for (MapFile.Reader r : readers) {
            //Get the first and last keys:
            long first = -1;
            long last = -1;

            //First key: no method for this for some inexplicable reason :/
            LongWritable k = new LongWritable();
            Writable v = ReflectionUtils.newInstance(valueClass, null);
            boolean hasNext = r.next(k, v);
            if(!hasNext){
                //This map file is empty - no data
                l.add(new Pair<>(-1L, -1L));
                continue;
            }
            first = k.get();

            //Last key: easy
            r.reset();
            r.finalKey(k);
            last = k.get();

            l.add(new Pair<>(first, last));
        }

        //Check that things are actually contiguous:
        List<Pair<Long, Long>> sorted = new ArrayList<>(l.size());
        for(Pair<Long,Long> p : l){
            if(p.getLeft() >= 0){
                sorted.add(p);
            }
        }
        Collections.sort(sorted, new Comparator<Pair<Long, Long>>() {
            @Override
            public int compare(Pair<Long, Long> o1, Pair<Long, Long> o2) {
                return Long.compare(o1.getFirst(), o2.getFirst());
            }
        });

        if (sorted.size() == 0){
            throw new IllegalStateException("Map file is empty - no data available");
        }
        if (sorted.get(0).getFirst() != 0L) {
            throw new UnsupportedOperationException("Minimum key value is not 0: got " + sorted.get(0).getFirst());
        }

        for (int i = 0; i < sorted.size() - 1; i++) {
            long currLast = sorted.get(i).getSecond();
            long nextFirst = sorted.get(i + 1).getFirst();

            if(nextFirst == -1){
                //Skip empty map file
                continue;
            }

            if (currLast + 1 != nextFirst) {
                throw new IllegalStateException(
                                "Keys are not contiguous between readers: first/last indices (inclusive) " + "are "
                                                + sorted
                                                + ".\n LongIndexKey assumes unique and contiguous LongWritable keys");
            }
        }

        readerIndices = l;
        return readerIndices;
    }

    @Override
    public LongWritable getKeyForIndex(long index) {
        return new LongWritable(index);
    }

    @Override
    public long getNumRecords() throws IOException {
        long max = -1;
        for (Pair<Long, Long> p : readerIndices) {
            max = Math.max(max, p.getSecond());
        }

        if (max <= 0) {
            throw new IllegalStateException("Invalid number of keys found: " + max);
        }

        return max + 1; //Zero indexed
    }
}
