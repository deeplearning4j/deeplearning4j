package org.deeplearning4j.plot;

import com.google.common.primitives.Ints;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Based on the work by krizshevy et. al
 *
 * http://nbviewer.ipython.org/github/udibr/caffe/blob/dev/examples/filter_visualization.ipynb
 *
 * @author Adam Gibson
 */
public class PlotFilters {
    /**
     * Render the given data
     * as filters
     * @param data the data to render
     * @param padSize the padding size for the images
     * @return the image to render
     */
    public INDArray render(INDArray data,int padSize) {
        data.subi(data.mean(Integer.MAX_VALUE));
        data.divi(data.max(Integer.MAX_VALUE));
        int n = (int) Math.ceil(Math.sqrt(Shape.squeeze(data.shape())[0]));

        int[][] padding = new int[4][];
        double end = Math.pow( n,2) - Shape.squeeze(data.shape())[0];
        padding[0] = new int[]{0,(int) end};
        padding[1] = new int[]{0,padSize};
        padding[2] = new int[]{0,padSize};
        List<double[]> list = new ArrayList<>();

        data = Nd4j.pad(data,padding, Nd4j.PadMode.CONSTANT);
        //# tile the filters into an image
        int[] baseFilterShape = Ints.concat(new int[]{n,n}, Arrays.copyOfRange(data.shape(),1,data.shape().length));
        data = data.reshape(baseFilterShape).permute(0, 2, 1, 3);
        if(data.rank() == 4)
            data = data.reshape(n * data.size(1),n * data.size(3));

        data.muli(255);

        return data;
    }


    /**
     * Render the given data
     * as filters
     * @param data the data to render
     * @param padSize the padding size for the images
     * @param padVal the value to pad with
     * @return the image to render
     */
    public INDArray render(INDArray data,int padSize,double padVal) {
        data.subi(data.mean(Integer.MAX_VALUE));
        data.divi(data.max(Integer.MAX_VALUE));
        // n = int(np.ceil(np.sqrt(data.shape[0])))
        int n = (int) Math.ceil(Math.sqrt(data.size(0)));
        int[][] padding = new int[4][];
        double end = Math.pow( n,2) - data.size(0);
        padding[0] = new int[]{0,(int) end};
        padding[1] = new int[]{0,padSize};
        padding[2] = new int[]{0,padSize};
        List<double[]> list = new ArrayList<>();

        data = Nd4j.pad(data,padding, Nd4j.PadMode.CONSTANT);
        //# tile the filters into an image
        int[] baseFilterShape = Ints.concat(new int[]{n,n}, Arrays.copyOfRange(data.shape(),1,data.shape().length));
        data = data.reshape(baseFilterShape).permute(0,2,1,3);
        data = data.reshape(n * data.size(1), n * data.size(3));
        data = Transforms.round(Transforms.abs(data)).muli(255);

        return data;
    }

}
