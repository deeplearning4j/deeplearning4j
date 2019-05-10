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
