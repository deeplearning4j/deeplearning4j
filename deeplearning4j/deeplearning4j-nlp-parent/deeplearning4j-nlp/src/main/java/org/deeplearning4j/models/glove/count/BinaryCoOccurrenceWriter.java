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

package org.deeplearning4j.models.glove.count;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;

/**
 * @author raver119@gmail.com
 */
public class BinaryCoOccurrenceWriter<T extends SequenceElement> implements CoOccurrenceWriter<T> {
    private File file;
    private DataOutputStream outputStream;

    private static final Logger log = LoggerFactory.getLogger(BinaryCoOccurrenceWriter.class);

    public BinaryCoOccurrenceWriter(@NonNull File file) {
        this.file = file;

        try {
            outputStream = new DataOutputStream(
                            new BufferedOutputStream(new FileOutputStream(file), 100 * 1024 * 1024));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void writeObject(@NonNull CoOccurrenceWeight<T> object) {
        try {
            //            log.info("Saving objects: { [" +object.getElement1().getIndex() +"], [" + object.getElement2().getIndex() + "]  }");
            outputStream.writeInt(object.getElement1().getIndex());
            outputStream.writeInt(object.getElement2().getIndex());
            outputStream.writeDouble(object.getWeight());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void queueObject(CoOccurrenceWeight<T> object) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void finish() {
        try {
            outputStream.flush();
        } catch (Exception e) {
        }

        try {
            outputStream.close();
        } catch (Exception e) {
        }
    }
}
