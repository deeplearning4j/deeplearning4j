package org.nd4j.jdbc.mysql;

import static org.junit.Assert.*;

import com.mchange.v2.c3p0.ComboPooledDataSource;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.sql.Blob;

public class MysqlLoaderTest {


    //simple litmus test, unfortunately relies on an external database
    @Test
    @Ignore
    public void testMysqlLoader() throws Exception {
        ComboPooledDataSource ds = new ComboPooledDataSource();
        ds.setJdbcUrl("jdbc:mysql://localhost:3306/nd4j?user=nd4j&password=nd4j");
        MysqlLoader loader = new MysqlLoader(ds,"jdbc:mysql://localhost:3306/nd4j?user=nd4j&password=nd4j","ndarrays","array");
        loader.delete("1");
        INDArray load = loader.load(loader.loadForID("1"));
        if(load != null) {
            loader.delete("1");
        }
        loader.save(Nd4j.create(new float[]{1,2,3}),"1");
        Blob b = loader.loadForID("1");
        INDArray loaded = loader.load(b);
        assertEquals((Nd4j.create(new float[]{1,2,3})),loaded);
    }

}