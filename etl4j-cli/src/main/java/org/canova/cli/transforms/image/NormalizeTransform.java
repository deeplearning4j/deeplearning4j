package org.canova.cli.transforms.image;

import java.util.Collection;
import java.util.Iterator;

import org.canova.api.io.data.DoubleWritable;
import org.canova.api.io.data.FloatWritable;
import org.canova.api.writable.Writable;
import org.canova.common.data.NDArrayWritable;
import org.canova.cli.transforms.Transform;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * For raw images like jpegs we need to perform transforms (normalize) 
 * - here we need to scan across the dataset first to get min / max 
 *
 * Since this is an image specific normalizer we find out min and max across all "columns" / pixels in the image collection
 * 		-	as opposed to just looking across columns between images in the colleciton
 * 		-	because pixel intensity is linked across the image grid of pixels
 *
 *
 * Questions:
 * 	-	are we able to do this in a way that will parallelize well later?
 * 			-	probably not, most likely requires a v2 refactor for MR
 *
 * Label Semantics
 * 		- NOTE: dont normalize the LABEL!
 * 			1.	Image: 	ImageInputFormat		> 	{ [array of doubles], directoryLabelID }		// image data, then the directory indexed as an ID int
 *
 *
 *
 * @author josh
 *
 */
public class NormalizeTransform implements Transform {

    public long totalRecords = 0;
    public double minValue = Double.NaN;
    public double maxValue = Double.NaN;

    /**
     * Transform a specific incoming vector in place
     *
     * TODO: is the label getting normalized here???
     *
     */
    @Override
    public void transform( Collection<Writable> vector ) {

        if (Double.NaN == this.minValue) {
            // throw exception?
            return;
        }

        Iterator<Writable> iter = vector.iterator();
        boolean isLabelEntry = false;
        // if we hit the last entry, its the label -- dont normalize it!
        if (!iter.hasNext()) {
            isLabelEntry = true;
        }
        while (iter.hasNext()) {

            Writable val =  iter.next();
            if(val instanceof NDArrayWritable) {
                INDArray arr = ((NDArrayWritable)val).get();
                for (int i = 0; i < arr.length(); i++) {
                    double range = this.maxValue - this.minValue;
                    double normalizedOut = ( arr.getDouble(i) - this.minValue ) / range;

                    if (0.0 == range) {
                        arr.putScalar(i, 0.0);
                    } else {
                        arr.putScalar(i, normalizedOut);
                    }
                }
            } else if(val instanceof DoubleWritable) {
                DoubleWritable dVal = (DoubleWritable) val;
                if (!isLabelEntry) {

                    double range = this.maxValue - this.minValue;
                    double normalizedOut = ( dVal.get() - this.minValue ) / range;

                    if (0.0 == range) {
                        dVal.set(0.0);
                    } else {
                        dVal.set(normalizedOut);
                    }

                }
            }
            else {
                FloatWritable valF = (FloatWritable) val;
                if (!isLabelEntry) {

                    float range = (float) (this.maxValue - this.minValue);
                    float normalizedOut = (float) (( valF.get() - this.minValue ) / range);

                    if (0.0 == range) {
                        valF.set(0.0f);
                    } else {
                        valF.set(normalizedOut);
                    }

                }
            }





        }





    }

    @Override
    public void collectStatistics(Collection<Writable> vector) {
        Iterator<Writable> iter = vector.iterator();
        double tmpVal;
        while (iter.hasNext()) {
            Writable val = iter.next();
            INDArray arr = val instanceof NDArrayWritable ? ((NDArrayWritable)val).get() : null;
            for (int i = 0; i < (arr != null ? arr.length() : 1); i++) {
                tmpVal = arr != null ? arr.getDouble(i) : Double.valueOf(val.toString());
                if (Double.isNaN( this.minValue)) {
                    this.minValue = tmpVal;

                } else if (tmpVal < this.minValue) {
                    this.minValue = tmpVal;
                }

                if (Double.isNaN( this.maxValue)) {
                    this.maxValue = tmpVal;

                } else if (tmpVal > this.maxValue) {
                    this.maxValue = tmpVal;
                }
            }

        }

    }

    @Override
    public void evaluateStatistics() {
        // TODO Auto-generated method stub

    }

}
