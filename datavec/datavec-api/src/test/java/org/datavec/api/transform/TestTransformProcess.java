/*
 *  * Copyright 2017 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.nlp.TextToCharacterIndexTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;

public class TestTransformProcess {

    @Test
    public void testExecution(){

        Schema schema = new Schema.Builder()
                .addColumnsString("col")
                .addColumnsDouble("col2")
                .build();

        Map<Character,Integer> m = defaultCharIndex();
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .doubleMathOp("col2", MathOp.Add, 1.0)
                .build();

        List<Writable> in = Arrays.<Writable>asList(new Text("Text"), new DoubleWritable(2.0));
        List<Writable> exp = Arrays.<Writable>asList(new Text("Text"), new DoubleWritable(3.0));

        List<Writable> out = transformProcess.execute(in);
        assertEquals(exp, out);
    }

    @Test
    public void testExecuteToSequence() {

        Schema schema = new Schema.Builder()
                .addColumnsString("action")
                .build();

        Map<Character,Integer> m = defaultCharIndex();
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .removeAllColumnsExceptFor("action")
                .convertToSequence()
                .transform(new TextToCharacterIndexTransform("action", "action_sequence", m, true))
                .build();

        String s = "in text";
        List<Writable> input = Collections.<Writable>singletonList(new Text(s));

        List<List<Writable>> expSeq = new ArrayList<>(s.length());
        for( int i = 0; i<s.length(); i++) {
            expSeq.add(Collections.<Writable>singletonList(new IntWritable(m.get(s.charAt(i)))));
        }


        List<List<Writable>> out = transformProcess.executeToSequence(input);

        assertEquals(expSeq, out);
    }

    @Test
    public void testInferColumns()  throws Exception {
        List<List<String>> categories = Arrays.asList(
                Arrays.asList("a","d")  ,
                Arrays.asList("b","e"),
                Arrays.asList("c","f")
        );

        RecordReader listReader = new ListStringRecordReader();
        listReader.initialize(new ListStringSplit(categories));
        List<String> inferredSingle = TransformProcess.inferCategories(listReader,0);
        assertEquals(3,inferredSingle.size());
        listReader.initialize(new ListStringSplit(categories));
        Map<Integer, List<String>> integerListMap = TransformProcess.inferCategories(listReader, new int[]{0,1});
        for(int i = 0; i < 2; i++) {
            assertEquals(3,integerListMap.get(i).size());
        }
    }


    public static Map<Character,Integer> defaultCharIndex() {
        Map<Character,Integer> ret = new TreeMap<>();

        ret.put('a',0);
        ret.put('b',1);
        ret.put('c',2);
        ret.put('d',3);
        ret.put('e',4);
        ret.put('f',5);
        ret.put('g',6);
        ret.put('h',7);
        ret.put('i',8);
        ret.put('j',9);
        ret.put('k',10);
        ret.put('l',11);
        ret.put('m',12);
        ret.put('n',13);
        ret.put('o',14);
        ret.put('p',15);
        ret.put('q',16);
        ret.put('r',17);
        ret.put('s',18);
        ret.put('t',19);
        ret.put('u',20);
        ret.put('v',21);
        ret.put('w',22);
        ret.put('x',23);
        ret.put('y',24);
        ret.put('z',25);
        ret.put('/',26);
        ret.put(' ',27);
        ret.put('(',28);
        ret.put(')',29);

        return ret;
    }



}
