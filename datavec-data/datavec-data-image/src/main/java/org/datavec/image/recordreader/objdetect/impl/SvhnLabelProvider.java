/*
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.image.recordreader.objdetect.impl;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;

import static org.bytedeco.javacpp.hdf5.*;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Label provider for object detection, for use with {@link org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader}.
 * This label provider reads the datasets from The Street View House Numbers (SVHN) Dataset.<br>
 * The SVHN datasets contain 10 classes (digits) with 73257 digits for training, 26032 digits for testing, and 531131 additional.<br>
 * <a href="http://ufldl.stanford.edu/housenumbers/">http://ufldl.stanford.edu/housenumbers/</a><br>
 * <br>
 * <br>
 * How to use:<br>
 * 1. Download and extract SVHN dataset with {@link org.deeplearning4j.datasets.fetchers.SvhnDataFetcher}<br>
 * 2. Set baseDirectory to (for example) "training" directory (should contain PNG images and a digitStruct.mat file)<br>
 *
 * @author saudet
 */
public class SvhnLabelProvider implements ImageObjectLabelProvider {

    private static DataType refType = new DataType(PredType.STD_REF_OBJ());
    private static DataType charType = new DataType(PredType.NATIVE_CHAR());
    private static DataType intType = new DataType(PredType.NATIVE_INT());

    private Map<String, List<ImageObject>> labelMap;

    public SvhnLabelProvider(File dir) throws IOException {
        labelMap = new HashMap<String, List<ImageObject>>();

        H5File file = new H5File(dir.getPath() + "/digitStruct.mat", H5F_ACC_RDONLY());
        Group group = file.openGroup("digitStruct");
        DataSet nameDataset = group.openDataSet("name");
        DataSpace nameSpace = nameDataset.getSpace();
        DataSet bboxDataset = group.openDataSet("bbox");
        DataSpace bboxSpace = bboxDataset.getSpace();
        long[] dims = new long[2];
        bboxSpace.getSimpleExtentDims(dims);
        int n = (int)(dims[0] * dims[1]);

        int ptrSize = Loader.sizeof(Pointer.class);
        PointerPointer namePtr = new PointerPointer(n);
        PointerPointer bboxPtr = new PointerPointer(n);
        nameDataset.read(namePtr, refType);
        bboxDataset.read(bboxPtr, refType);

        BytePointer bytePtr = new BytePointer(256);
        PointerPointer topPtr = new PointerPointer(256);
        PointerPointer leftPtr = new PointerPointer(256);
        PointerPointer heightPtr = new PointerPointer(256);
        PointerPointer widthPtr = new PointerPointer(256);
        PointerPointer labelPtr = new PointerPointer(256);
        IntPointer intPtr = new IntPointer(256);
        for (int i = 0; i < n; i++) {
            DataSet nameRef = new DataSet(file, namePtr.position(i * ptrSize));
            nameRef.read(bytePtr, charType);
            String filename = bytePtr.getString();

            Group bboxGroup = new Group(file, bboxPtr.position(i * ptrSize));
            DataSet topDataset = bboxGroup.openDataSet("top");
            DataSet leftDataset = bboxGroup.openDataSet("left");
            DataSet heightDataset = bboxGroup.openDataSet("height");
            DataSet widthDataset = bboxGroup.openDataSet("width");
            DataSet labelDataset = bboxGroup.openDataSet("label");

            DataSpace topSpace = topDataset.getSpace();
            topSpace.getSimpleExtentDims(dims);
            int m = (int)(dims[0] * dims[1]);
            ArrayList<ImageObject> list = new ArrayList<ImageObject>(m);

            boolean isFloat = topDataset.asAbstractDs().getTypeClass() == H5T_FLOAT;
            if (!isFloat) {
                topDataset.read(topPtr.position(0), refType);
                leftDataset.read(leftPtr.position(0), refType);
                heightDataset.read(heightPtr.position(0), refType);
                widthDataset.read(widthPtr.position(0), refType);
                labelDataset.read(labelPtr.position(0), refType);
            }
            assert !isFloat || m == 1;

            for (int j = 0; j < m; j++) {
                DataSet topSet = isFloat ? topDataset : new DataSet(file, topPtr.position(j * ptrSize));
                topSet.read(intPtr, intType);
                int top = intPtr.get();

                DataSet leftSet = isFloat ? leftDataset : new DataSet(file, leftPtr.position(j * ptrSize));
                leftSet.read(intPtr, intType);
                int left = intPtr.get();

                DataSet heightSet = isFloat ? heightDataset : new DataSet(file, heightPtr.position(j * ptrSize));
                heightSet.read(intPtr, intType);
                int height = intPtr.get();

                DataSet widthSet = isFloat ? widthDataset : new DataSet(file, widthPtr.position(j * ptrSize));
                widthSet.read(intPtr, intType);
                int width = intPtr.get();

                DataSet labelSet = isFloat ? labelDataset : new DataSet(file, labelPtr.position(j * ptrSize));
                labelSet.read(intPtr, intType);
                int label = intPtr.get();
                if (label == 10) {
                    label = 0;
                }

                list.add(new ImageObject(left, top, left + width, top + height, Integer.toString(label)));

                topSet.deallocate();
                leftSet.deallocate();
                heightSet.deallocate();
                widthSet.deallocate();
                labelSet.deallocate();
            }

            topSpace.deallocate();
            if (!isFloat) {
                topDataset.deallocate();
                leftDataset.deallocate();
                heightDataset.deallocate();
                widthDataset.deallocate();
                labelDataset.deallocate();
            }
            nameRef.deallocate();
            bboxGroup.deallocate();

            labelMap.put(filename, list);
        }

        nameSpace.deallocate();
        bboxSpace.deallocate();
        nameDataset.deallocate();
        bboxDataset.deallocate();
        group.deallocate();
        file.deallocate();
    }

    @Override
    public List<ImageObject> getImageObjectsForPath(String path) {
        File file = new File(path);
        String filename = file.getName();
        return labelMap.get(filename);
    }

    @Override
    public List<ImageObject> getImageObjectsForPath(URI uri) {
        return getImageObjectsForPath(uri.toString());
    }
}
