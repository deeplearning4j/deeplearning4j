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

package org.datavec.api.transform.reduce;

import org.datavec.api.transform.ops.IAggregableReduceOp;
import org.datavec.api.transform.reduce.impl.GeographicMidpointReduction;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TestReductions {

    @Test
    public void testGeographicMidPointReduction(){

        //http://www.geomidpoint.com/example.html
        //That particular example is weighted - have 3x weight for t1, 2x weight for t2, 1x weight for t1
        Text t1 = new Text("40.7143528,-74.0059731");
        Text t2 = new Text("41.8781136,-87.6297982");
        Text t3 = new Text("33.7489954,-84.3879824");

        List<Writable> list = Arrays.<Writable>asList(t1, t1, t1, t2, t2, t3);

        GeographicMidpointReduction reduction = new GeographicMidpointReduction(",");

        IAggregableReduceOp<Writable, List<Writable>> reduceOp = reduction.reduceOp();
        for(Writable w : list){
            reduceOp.accept(w);
        }

        List<Writable> listOut = reduceOp.get();
        assertEquals(1, listOut.size());
        Writable wOut = listOut.get(0);

        String[] split = wOut.toString().split(",");
        double lat = Double.parseDouble(split[0]);
        double lng = Double.parseDouble(split[1]);

        double expLat = 40.11568861;
        double expLong = -80.29960280;

        assertEquals(expLat, lat, 1e-6);
        assertEquals(expLong, lng, 1e-6);


        //Test multiple reductions
        list = Arrays.<Writable>asList(t1, t1, t2);
        List<Writable> list2 = Arrays.<Writable>asList(t1, t2, t3);
        reduceOp = reduction.reduceOp();
        for(Writable w : list){
            reduceOp.accept(w);
        }

        IAggregableReduceOp<Writable, List<Writable>> reduceOp2 = reduction.reduceOp();
        for(Writable w : list2){
            reduceOp2.accept(w);
        }

        reduceOp.combine(reduceOp2);

        listOut = reduceOp.get();
        assertEquals(1, listOut.size());
        wOut = listOut.get(0);

        split = wOut.toString().split(",");
        lat = Double.parseDouble(split[0]);
        lng = Double.parseDouble(split[1]);

        assertEquals(expLat, lat, 1e-6);
        assertEquals(expLong, lng, 1e-6);
    }

}
