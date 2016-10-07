package org.deeplearning4j.ui;

import org.deeplearning4j.ui.storage.StorageMetaData;
import org.junit.Test;

import java.io.Serializable;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 07/10/2016.
 */
public class TestStorageMetaData {

    @Test
    public void testStorageMetaData(){

        Serializable extraMeta = "ExtraMetaData";
        long timeStamp = 123456;
        StorageMetaData m = new StorageMetaData( timeStamp,
                "sessionID", "typeID", "workerID", "org.some.class.InitType", "org.some.class.UpdateType", extraMeta);

        byte[] bytes = m.encode();
        StorageMetaData m2 = new StorageMetaData();
        m2.decode(bytes);

        assertEquals(m, m2);
        assertArrayEquals(bytes, m2.encode());

        //Sanity check: null values
        m = new StorageMetaData(0, null,null,null,null,(String)null);
        bytes = m.encode();
        m2 = new StorageMetaData();
        m2.decode(bytes);
        assertEquals(m, m2);
        assertArrayEquals(bytes, m2.encode());
    }

}
