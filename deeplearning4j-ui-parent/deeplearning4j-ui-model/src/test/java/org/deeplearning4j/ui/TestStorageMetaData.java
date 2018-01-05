package org.deeplearning4j.ui;

import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.ui.storage.impl.SbeStorageMetaData;
import org.junit.Test;

import java.io.Serializable;

import static org.junit.Assert.*;

/**
 * Created by Alex on 07/10/2016.
 */
public class TestStorageMetaData {

    @Test
    public void testStorageMetaData() {

        Serializable extraMeta = "ExtraMetaData";
        long timeStamp = 123456;
        StorageMetaData m = new SbeStorageMetaData(timeStamp, "sessionID", "typeID", "workerID",
                        "org.some.class.InitType", "org.some.class.UpdateType", extraMeta);

        byte[] bytes = m.encode();
        StorageMetaData m2 = new SbeStorageMetaData();
        m2.decode(bytes);

        assertEquals(m, m2);
        assertArrayEquals(bytes, m2.encode());

        //Sanity check: null values
        m = new SbeStorageMetaData(0, null, null, null, null, (String) null);
        bytes = m.encode();
        m2 = new SbeStorageMetaData();
        m2.decode(bytes);
        //In practice, we don't want these things to ever be null anyway...
        assertNullOrZeroLength(m2.getSessionID());
        assertNullOrZeroLength(m2.getTypeID());
        assertNullOrZeroLength(m2.getWorkerID());
        assertNullOrZeroLength(m2.getInitTypeClass());
        assertNullOrZeroLength(m2.getUpdateTypeClass());
        assertArrayEquals(bytes, m2.encode());
    }

    private static void assertNullOrZeroLength(String str) {
        assertTrue(str == null || str.length() == 0);
    }

}
