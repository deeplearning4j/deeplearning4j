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

package org.deeplearning4j.ui.weights;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.api.storage.Persistable;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.ByteBuffer;

/**
 * Created by Alex on 24/10/2016.
 */
@AllArgsConstructor
@Data
public class ConvolutionListenerPersistable implements Persistable {

    private static final String TYPE_ID = "ConvolutionalListener";

    private String sessionID;
    private String workerID;
    private long timestamp;
    private transient BufferedImage img;

    public ConvolutionListenerPersistable() {}

    @Override
    public String getSessionID() {
        return sessionID;
    }

    @Override
    public String getTypeID() {
        return TYPE_ID;
    }

    @Override
    public String getWorkerID() {
        return workerID;
    }

    @Override
    public long getTimeStamp() {
        return timestamp;
    }

    @Override
    public int encodingLengthBytes() {
        return 0;
    }

    @Override
    public byte[] encode() {
        //Not the most efficient: but it's easy to implement...
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(this);
        } catch (IOException e) {
            throw new RuntimeException(e); //Shouldn't normally happen
        }

        return baos.toByteArray();
    }

    @Override
    public void encode(ByteBuffer buffer) {
        buffer.put(encode());
    }

    @Override
    public void encode(OutputStream outputStream) throws IOException {
        outputStream.write(encode());
    }

    @Override
    public void decode(byte[] decode) {
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(decode))) {
            ConvolutionListenerPersistable p = (ConvolutionListenerPersistable) ois.readObject();
            this.sessionID = p.sessionID;
            this.workerID = p.workerID;
            this.timestamp = p.getTimeStamp();
            this.img = p.img;
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e); //Shouldn't normally happen
        }
    }

    @Override
    public void decode(ByteBuffer buffer) {
        byte[] arr = new byte[buffer.remaining()];
        buffer.get(arr);
        decode(arr);
    }

    @Override
    public void decode(InputStream inputStream) throws IOException {
        byte[] b = IOUtils.toByteArray(inputStream);
        decode(b);
    }

    private void writeObject(ObjectOutputStream oos) throws IOException {
        oos.defaultWriteObject();
        ImageIO.write(img, "png", oos);
    }

    private void readObject(ObjectInputStream ois) throws IOException, ClassNotFoundException {
        ois.defaultReadObject();
        img = ImageIO.read(ois);
    }
}
