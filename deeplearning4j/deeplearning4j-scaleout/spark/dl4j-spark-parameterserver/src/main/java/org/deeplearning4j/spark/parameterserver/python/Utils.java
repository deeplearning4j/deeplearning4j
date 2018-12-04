package org.deeplearning4j.spark.parameterserver.python;

import org.apache.spark.api.java.JavaRDD;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import javax.xml.crypto.Data;

public class Utils {

    public static JavaRDD<ArrayDescriptor> getArrayDescriptorRDD(JavaRDD<INDArray> indarrayRDD){
        return indarrayRDD.map(arr -> new ArrayDescriptor(arr));
    }

    public static  JavaRDD<INDArray> getArrayRDD(JavaRDD<ArrayDescriptor> arrayDescriptorRDD){
        return arrayDescriptorRDD.map(ad -> ad.getArray());
    }

    public static JavaRDD<DataSetDescriptor> getDatasetDescriptorRDD(JavaRDD<DataSet> dsRDD){
        return dsRDD.map(ds -> new DataSetDescriptor(ds));
    }

    public static JavaRDD<DataSet> getDataSetRDD(JavaRDD<DataSetDescriptor> dsDescriptorRDD){
        return dsDescriptorRDD.map(dsd -> dsd.getDataSet());
    }
}
