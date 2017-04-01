/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 *
 */

package org.nd4j.jdbc.mysql;

import com.mchange.v2.c3p0.ComboPooledDataSource;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.sql.Blob;

import static org.junit.Assert.assertEquals;

public class MysqlLoaderTest {


    //simple litmus test, unfortunately relies on an external database
    @Test
    @Ignore
    public void testMysqlLoader() throws Exception {
        ComboPooledDataSource ds = new ComboPooledDataSource();
        ds.setJdbcUrl("jdbc:mysql://localhost:3306/nd4j?user=nd4j&password=nd4j");
        MysqlLoader loader = new MysqlLoader(ds, "jdbc:mysql://localhost:3306/nd4j?user=nd4j&password=nd4j", "ndarrays",
                        "array");
        loader.delete("1");
        INDArray load = loader.load(loader.loadForID("1"));
        if (load != null) {
            loader.delete("1");
        }
        loader.save(Nd4j.create(new float[] {1, 2, 3}), "1");
        Blob b = loader.loadForID("1");
        INDArray loaded = loader.load(b);
        assertEquals((Nd4j.create(new float[] {1, 2, 3})), loaded);
    }

}
