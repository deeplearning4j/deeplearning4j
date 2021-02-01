/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.hdf5.*;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.Closeable;
import java.io.IOException;
import java.lang.Exception;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.hdf5.global.hdf5.*;

@Slf4j
public class Hdf5Archive implements Closeable {

    public static final int MAX_BUFFER_SIZE_BYTES = (int)Math.pow(2, 28);       //256 MB

    /**
     * HDF5 library is not thread safe - possible to crash if multiple reads etc are performed concurrently
     * in multiple threads. This object is used for locking read etc activity using synchronized blocks
     */
    public static final Object LOCK_OBJECT = new Object();

    static {
        try {
            /* This is necessary for the call to the BytePointer constructor below. */
            Loader.load(org.bytedeco.hdf5.global.hdf5.class);
        } catch (Exception e) {
            log.error("",e);
        }
    }

    private H5File file;
    private static DataType dataType = new DataType(PredType.NATIVE_FLOAT());

    public Hdf5Archive(String archiveFilename) {
        synchronized (LOCK_OBJECT) {
            this.file = new H5File(archiveFilename, H5F_ACC_RDONLY());
        }
    }

    @Override public void close() {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            file.deallocate();
        }
    }

    private Group[] openGroups(String... groups) {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            Group[] groupArray = new Group[groups.length];
            groupArray[0] = this.file.openGroup(groups[0]);
            for (int i = 1; i < groups.length; i++) {
                groupArray[i] = groupArray[i - 1].openGroup(groups[i]);
            }
            return groupArray;
        }
    }

    private void closeGroups(Group[] groupArray) {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            for (int i = groupArray.length - 1; i >= 0; i--) {
                groupArray[i].deallocate();
            }
        }
    }

    /**
     * Read data set as ND4J array from group path.
     *
     * @param datasetName Name of data set
     * @param groups      Array of zero or more ancestor groups from root to parent.
     * @return INDArray of HDF5 group data
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public INDArray readDataSet(String datasetName, String... groups) throws UnsupportedKerasConfigurationException {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            if (groups.length == 0)
                return readDataSet(this.file, datasetName);
            Group[] groupArray = openGroups(groups);
            INDArray a = readDataSet(groupArray[groupArray.length - 1], datasetName);
            closeGroups(groupArray);
            return a;
        }
    }

    /**
     * Read JSON-formatted string attribute from group path.
     *
     * @param attributeName Name of attribute
     * @param groups        Array of zero or more ancestor groups from root to parent.
     * @return HDF5 attribute as JSON
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public String readAttributeAsJson(String attributeName, String... groups)
            throws UnsupportedKerasConfigurationException {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            if (groups.length == 0) {
                Attribute a = this.file.openAttribute(attributeName);
                String s = readAttributeAsJson(a);
                a.deallocate();
                return s;
            }
            Group[] groupArray = openGroups(groups);
            Attribute a = groupArray[groups.length - 1].openAttribute(attributeName);
            String s = readAttributeAsJson(a);
            a.deallocate();
            closeGroups(groupArray);
            return s;
        }
    }

    /**
     * Read string attribute from group path.
     *
     * @param attributeName Name of attribute
     * @param groups        Array of zero or more ancestor groups from root to parent.
     * @return HDF5 attribute as String
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public String readAttributeAsString(String attributeName, String... groups)
            throws UnsupportedKerasConfigurationException {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            if (groups.length == 0) {
                Attribute a = this.file.openAttribute(attributeName);
                String s = readAttributeAsString(a);
                a.deallocate();
                return s;
            }
            Group[] groupArray = openGroups(groups);
            Attribute a = groupArray[groups.length - 1].openAttribute(attributeName);
            String s = readAttributeAsString(a);
            a.deallocate();
            closeGroups(groupArray);
            return s;
        }
    }

    /**
     * Check whether group path contains string attribute.
     *
     * @param attributeName Name of attribute
     * @param groups        Array of zero or more ancestor groups from root to parent.
     * @return Boolean indicating whether attribute exists in group path.
     */
    public boolean hasAttribute(String attributeName, String... groups) {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            if (groups.length == 0)
                return this.file.attrExists(attributeName);
            Group[] groupArray = openGroups(groups);
            boolean b = groupArray[groupArray.length - 1].attrExists(attributeName);
            closeGroups(groupArray);
            return b;
        }
    }

    /**
     * Get list of data sets from group path.
     *
     * @param groups Array of zero or more ancestor groups from root to parent.
     * @return List of HDF5 data set names
     */
    public List<String> getDataSets(String... groups) {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            if (groups.length == 0)
                return getObjects(this.file, H5O_TYPE_DATASET);
            Group[] groupArray = openGroups(groups);
            List<String> ls = getObjects(groupArray[groupArray.length - 1], H5O_TYPE_DATASET);
            closeGroups(groupArray);
            return ls;
        }
    }

    /**
     * Get list of groups from group path.
     *
     * @param groups Array of zero or more ancestor groups from root to parent.
     * @return List of HDF5 groups
     */
    public List<String> getGroups(String... groups) {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            if (groups.length == 0)
                return getObjects(this.file, H5O_TYPE_GROUP);
            Group[] groupArray = openGroups(groups);
            List<String> ls = getObjects(groupArray[groupArray.length - 1], H5O_TYPE_GROUP);
            closeGroups(groupArray);
            return ls;
        }
    }

    /**
     * Read data set as ND4J array from HDF5 group.
     *
     * @param fileGroup   HDF5 file or group
     * @param datasetName Name of data set
     * @return INDArray from HDF5 data set
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    private INDArray readDataSet(Group fileGroup, String datasetName)
            throws UnsupportedKerasConfigurationException {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            DataSet dataset = fileGroup.openDataSet(datasetName);
            DataSpace space = dataset.getSpace();
            int nbDims = space.getSimpleExtentNdims();
            long[] dims = new long[nbDims];
            space.getSimpleExtentDims(dims);
            float[] dataBuffer;
            FloatPointer fp;
            int j;
            INDArray data;
            switch (nbDims) {
                case 5: /* 3D Convolution weights */
                    dataBuffer = new float[(int) (dims[0] * dims[1] * dims[2] * dims[3] * dims[4])];
                    fp = new FloatPointer(dataBuffer);
                    dataset.read(fp, dataType);
                    fp.get(dataBuffer);
                    data = Nd4j.create((int) dims[0], (int) dims[1], (int) dims[2], (int) dims[3], (int) dims[4]);
                    j = 0;
                    for (int i1 = 0; i1 < dims[0]; i1++)
                        for (int i2 = 0; i2 < dims[1]; i2++)
                            for (int i3 = 0; i3 < dims[2]; i3++)
                                for (int i4 = 0; i4 < dims[3]; i4++)
                                    for (int i5 = 0; i5 < dims[4]; i5++)
                                        data.putScalar(new int[] { i1, i2, i3, i4, i5 }, dataBuffer[j++]);
                    break;
                case 4: /* 2D Convolution weights */
                    dataBuffer = new float[(int) (dims[0] * dims[1] * dims[2] * dims[3])];
                    fp = new FloatPointer(dataBuffer);
                    dataset.read(fp, dataType);
                    fp.get(dataBuffer);
                    data = Nd4j.create((int) dims[0], (int) dims[1], (int) dims[2], (int) dims[3]);
                    j = 0;
                    for (int i1 = 0; i1 < dims[0]; i1++)
                        for (int i2 = 0; i2 < dims[1]; i2++)
                            for (int i3 = 0; i3 < dims[2]; i3++)
                                for (int i4 = 0; i4 < dims[3]; i4++)
                                    data.putScalar(i1, i2, i3, i4, dataBuffer[j++]);
                    break;
                case 3:
                    dataBuffer = new float[(int) (dims[0] * dims[1] * dims[2])];
                    fp = new FloatPointer(dataBuffer);
                    dataset.read(fp, dataType);
                    fp.get(dataBuffer);
                    data = Nd4j.create((int) dims[0], (int) dims[1], (int) dims[2]);
                    j = 0;
                    for (int i1 = 0; i1 < dims[0]; i1++)
                        for (int i2 = 0; i2 < dims[1]; i2++)
                            for (int i3 = 0; i3 < dims[2]; i3++)
                                data.putScalar(i1, i2, i3, dataBuffer[j++]);
                    break;
                case 2: /* Dense and Recurrent weights */
                    dataBuffer = new float[(int) (dims[0] * dims[1])];
                    fp = new FloatPointer(dataBuffer);
                    dataset.read(fp, dataType);
                    fp.get(dataBuffer);
                    data = Nd4j.create((int) dims[0], (int) dims[1]);
                    j = 0;
                    for (int i1 = 0; i1 < dims[0]; i1++)
                        for (int i2 = 0; i2 < dims[1]; i2++)
                            data.putScalar(i1, i2, dataBuffer[j++]);
                    break;
                case 1: /* Bias */
                    dataBuffer = new float[(int) dims[0]];
                    fp = new FloatPointer(dataBuffer);
                    dataset.read(fp, dataType);
                    fp.get(dataBuffer);
                    data = Nd4j.create((int) dims[0]);
                    j = 0;
                    for (int i1 = 0; i1 < dims[0]; i1++)
                        data.putScalar(i1, dataBuffer[j++]);
                    break;
                default:
                    throw new UnsupportedKerasConfigurationException("Cannot import weights with rank " + nbDims);
            }
            space.deallocate();
            dataset.deallocate();
            return data;
        }
    }

    /**
     * Get list of objects with a given type from a file group.
     *
     * @param fileGroup HDF5 file or group
     * @param objType   Type of object as integer
     * @return List of HDF5 group objects
     */
    private List<String> getObjects(Group fileGroup, int objType) {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            List<String> groups = new ArrayList<>();
            for (int i = 0; i < fileGroup.getNumObjs(); i++) {
                BytePointer objPtr = fileGroup.getObjnameByIdx(i);
                if (fileGroup.childObjType(objPtr) == objType)
                    groups.add(fileGroup.getObjnameByIdx(i).getString());
            }
            return groups;
        }
    }

    /**
     * Read JSON-formatted string attribute.
     *
     * @param attribute HDF5 attribute to read as JSON formatted string.
     * @return JSON formatted string from HDF5 attribute
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    private String readAttributeAsJson(Attribute attribute) throws UnsupportedKerasConfigurationException {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            VarLenType vl = attribute.getVarLenType();
            int currBufferLength = 2048;
            String s;
            /* TODO: find a less hacky way to do this.
             * Reading variable length strings (from attributes) is a giant
             * pain. There does not appear to be any way to determine the
             * length of the string in advance, so we use a hack: choose a
             * buffer size and read the config. If Jackson fails to parse
             * it, then we must not have read the entire config. Increase
             * buffer and repeat.
             */
            while (true) {
                byte[] attrBuffer = new byte[currBufferLength];
                BytePointer attrPointer = new BytePointer(currBufferLength);
                attribute.read(vl, attrPointer);
                attrPointer.get(attrBuffer);
                s = new String(attrBuffer);
                ObjectMapper mapper = new ObjectMapper();
                mapper.enable(DeserializationFeature.FAIL_ON_READING_DUP_TREE_KEY);
                try {
                    mapper.readTree(s);
                    break;
                } catch (IOException e) {
                    //OK - we don't know how long the buffer needs to be, so we'll try again with larger buffer
                }

                if(currBufferLength == MAX_BUFFER_SIZE_BYTES){
                    throw new UnsupportedKerasConfigurationException("Could not read abnormally long HDF5 attribute: size exceeds " + currBufferLength + " bytes");
                } else {
                    currBufferLength = (int)Math.min(MAX_BUFFER_SIZE_BYTES, currBufferLength * 4L);
                }
            }
            vl.deallocate();
            return s;
        }
    }

    /**
     * Read attribute as string.
     *
     * @param attribute HDF5 attribute to read as string.
     * @return HDF5 attribute as string
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    private String readAttributeAsString(Attribute attribute) throws UnsupportedKerasConfigurationException {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            VarLenType vl = attribute.getVarLenType();
            int bufferSizeMult = 1;
            String s = null;
            /* TODO: find a less hacky way to do this.
             * Reading variable length strings (from attributes) is a giant
             * pain. There does not appear to be any way to determine the
             * length of the string in advance, so we use a hack: choose a
             * buffer size and read the config, increase buffer and repeat
             * until the buffer ends with \u0000
             */
            while (true) {
                byte[] attrBuffer = new byte[bufferSizeMult * 2000];
                BytePointer attrPointer = new BytePointer(attrBuffer);
                attribute.read(vl, attrPointer);
                attrPointer.get(attrBuffer);
                s = new String(attrBuffer);

                if (s.endsWith("\u0000")) {
                    s = s.replace("\u0000", "");
                    break;
                }

                bufferSizeMult++;
                if (bufferSizeMult > 1000) {
                    throw new UnsupportedKerasConfigurationException("Could not read abnormally long HDF5 attribute");
                }
            }
            vl.deallocate();
            return s;
        }
    }

    /**
     * Read string attribute from group path.
     *
     * @param attributeName Name of attribute
     * @param bufferSize    buffer size to read
     * @return Fixed-length string read from HDF5 attribute name
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public String readAttributeAsFixedLengthString(String attributeName, int bufferSize)
            throws UnsupportedKerasConfigurationException {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            Attribute a = this.file.openAttribute(attributeName);
            String s = readAttributeAsFixedLengthString(a, bufferSize);
            a.deallocate();
            return s;
        }
    }

    /**
     * Read attribute of fixed buffer size as string.
     *
     * @param attribute HDF5 attribute to read as string.
     * @return Fixed-length string read from HDF5 attribute
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    private String readAttributeAsFixedLengthString(Attribute attribute, int bufferSize)
            throws UnsupportedKerasConfigurationException {
        synchronized (Hdf5Archive.LOCK_OBJECT) {
            VarLenType vl = attribute.getVarLenType();
            byte[] attrBuffer = new byte[bufferSize];
            BytePointer attrPointer = new BytePointer(attrBuffer);
            attribute.read(vl, attrPointer);
            attrPointer.get(attrBuffer);
            vl.deallocate();
            return new String(attrBuffer);
        }
    }
}
