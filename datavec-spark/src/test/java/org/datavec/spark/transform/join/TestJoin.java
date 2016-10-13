/*
 *  * Copyright 2016 Skymind, Inc.
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

package org.datavec.spark.transform.join;

import org.apache.spark.api.java.JavaRDD;
import org.datavec.api.transform.join.Join;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.BaseSparkTest;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 13/10/2016.
 */
public class TestJoin extends BaseSparkTest {

    @Test
    public void testJoinOneToMany_ManyToOne(){

        Schema customerInfoSchema = new Schema.Builder()
                .addColumnLong("customerID")
                .addColumnString("customerName")
                .build();

        Schema purchasesSchema = new Schema.Builder()
                .addColumnString("purchaseID")
                .addColumnsInteger("customerID")
                .addColumnDouble("amount")
                .build();

        List<List<Writable>> infoList = new ArrayList<>();
        infoList.add(Arrays.<Writable>asList(new LongWritable(12345),new Text("Customer12345")));
        infoList.add(Arrays.<Writable>asList(new LongWritable(98765),new Text("Customer98765")));
        infoList.add(Arrays.<Writable>asList(new LongWritable(50000),new Text("Customer50000")));

        List<List<Writable>> purchaseList = new ArrayList<>();
        purchaseList.add(Arrays.<Writable>asList(new LongWritable(1000000),new LongWritable(12345), new DoubleWritable(10.00)));
        purchaseList.add(Arrays.<Writable>asList(new LongWritable(1000001),new LongWritable(12345), new DoubleWritable(20.00)));
        purchaseList.add(Arrays.<Writable>asList(new LongWritable(1000002),new LongWritable(98765), new DoubleWritable(30.00)));

        Join join = new Join.Builder(Join.JoinType.RightOuter)
                .setKeyColumns("customerID")
                .setSchemas(customerInfoSchema, purchasesSchema)
                .build();

        List<List<Writable>> expected = new ArrayList<>();
        expected.add(Arrays.<Writable>asList(new LongWritable(12345),new Text("Customer12345"), new LongWritable(1000000), new DoubleWritable(10.00)));
        expected.add(Arrays.<Writable>asList(new LongWritable(12345),new Text("Customer12345"), new LongWritable(1000001), new DoubleWritable(20.00)));
        expected.add(Arrays.<Writable>asList(new LongWritable(98765),new Text("Customer98765"), new LongWritable(1000002), new DoubleWritable(30.00)));


        JavaRDD<List<Writable>> info = sc.parallelize(infoList);
        JavaRDD<List<Writable>> purchases = sc.parallelize(purchaseList);

        JavaRDD<List<Writable>> joined = SparkTransformExecutor.executeJoin(join, info, purchases);
        List<List<Writable>> joinedList = joined.collect();

        assertEquals(3, joinedList.size());


        //Test Many to one: same thing, but swap the order...
        Join join2 = new Join.Builder(Join.JoinType.LeftOuter)
                .setKeyColumns("customerID")
                .setSchemas(purchasesSchema, customerInfoSchema)
                .build();

        List<List<Writable>> expectedManyToOne = new ArrayList<>();
        expectedManyToOne.add(Arrays.<Writable>asList(new LongWritable(1000000),new LongWritable(12345), new DoubleWritable(10.00), new Text("Customer12345")));
        expectedManyToOne.add(Arrays.<Writable>asList(new LongWritable(1000001),new LongWritable(12345), new DoubleWritable(20.00), new Text("Customer12345")));
        expectedManyToOne.add(Arrays.<Writable>asList(new LongWritable(1000002),new LongWritable(98765), new DoubleWritable(30.00), new Text("Customer98765")));

        JavaRDD<List<Writable>> joined2 = SparkTransformExecutor.executeJoin(join2, purchases, info);
        List<List<Writable>> joinedList2 = joined2.collect();
        assertEquals(3, joinedList2.size());

        
    }

}
