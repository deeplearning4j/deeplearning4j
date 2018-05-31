package org.deeplearning4j.nn.layers.objdetect;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import static org.nd4j.linalg.indexing.NDArrayIndex.*;

/**
 * Functionality to interpret the network output of Yolo2OutputLayer.
 *
 * @author saudet
 */
public class YoloUtils {

    /** Essentially: just apply activation functions... */
    public static INDArray activate(INDArray boundingBoxPriors, INDArray input) {
        return activate(boundingBoxPriors, input, LayerWorkspaceMgr.noWorkspaces());
    }

    public static INDArray activate(INDArray boundingBoxPriors, INDArray input, LayerWorkspaceMgr layerWorkspaceMgr){
        // FIXME: int cast
        int mb = (int) input.size(0);
        int h = (int) input.size(2);
        int w = (int) input.size(3);
        int b = (int) boundingBoxPriors.size(0);
        int c = (int) (input.size(1)/b)-5;  //input.size(1) == b * (5 + C) -> C = (input.size(1)/b) - 5

        INDArray output = layerWorkspaceMgr.create(ArrayType.ACTIVATIONS, input.shape(), 'c');
        INDArray output5 = output.reshape('c', mb, b, 5+c, h, w);
        INDArray output4 = output;  //output.get(all(), interval(0,5*b), all(), all());
        INDArray input4 = input.dup('c');    //input.get(all(), interval(0,5*b), all(), all()).dup('c');
        INDArray input5 = input4.reshape('c', mb, b, 5+c, h, w);

        //X/Y center in grid: sigmoid
        INDArray predictedXYCenterGrid = input5.get(all(), all(), interval(0,2), all(), all());
        Transforms.sigmoid(predictedXYCenterGrid, false);

        //width/height: prior * exp(input)
        INDArray predictedWHPreExp = input5.get(all(), all(), interval(2,4), all(), all());
        INDArray predictedWH = Transforms.exp(predictedWHPreExp, false);
        Broadcast.mul(predictedWH, boundingBoxPriors, predictedWH, 1, 2);  //Box priors: [b, 2]; predictedWH: [mb, b, 2, h, w]

        //Confidence - sigmoid
        INDArray predictedConf = input5.get(all(), all(), point(4), all(), all());   //Shape: [mb, B, H, W]
        Transforms.sigmoid(predictedConf, false);

        output4.assign(input4);

        //Softmax
        //TODO OPTIMIZE?
        INDArray inputClassesPreSoftmax = input5.get(all(), all(), interval(5, 5+c), all(), all());   //Shape: [minibatch, C, H, W]
        INDArray classPredictionsPreSoftmax2d = inputClassesPreSoftmax.permute(0,1,3,4,2) //[minibatch, b, c, h, w] To [mb, b, h, w, c]
                .dup('c').reshape('c', new int[]{mb*b*h*w, c});
        Transforms.softmax(classPredictionsPreSoftmax2d, false);
        INDArray postSoftmax5d = classPredictionsPreSoftmax2d.reshape('c', mb, b, h, w, c ).permute(0, 1, 4, 2, 3);

        INDArray outputClasses = output5.get(all(), all(), interval(5, 5+c), all(), all());   //Shape: [minibatch, C, H, W]
        outputClasses.assign(postSoftmax5d);

        return output;
    }

    /** Returns overlap between lines [x1, x2] and [x3. x4]. */
    public static double overlap(double x1, double x2, double x3, double x4) {
        if (x3 < x1) {
            if (x4 < x1) {
                return 0;
            } else {
                return Math.min(x2, x4) - x1;
            }
        } else {
            if (x2 < x3) {
                return 0;
            } else {
                return Math.min(x2, x4) - x3;
            }
        }
    }

    /** Returns intersection over union (IOU) between o1 and o2. */
    public static double iou(DetectedObject o1, DetectedObject o2) {
        double x1min  = o1.getCenterX() - o1.getWidth() / 2;
        double x1max  = o1.getCenterX() + o1.getWidth() / 2;
        double y1min  = o1.getCenterY() - o1.getHeight() / 2;
        double y1max  = o1.getCenterY() + o1.getHeight() / 2;

        double x2min  = o2.getCenterX() - o2.getWidth() / 2;
        double x2max  = o2.getCenterX() + o2.getWidth() / 2;
        double y2min  = o2.getCenterY() - o2.getHeight() / 2;
        double y2max  = o2.getCenterY() + o2.getHeight() / 2;

        double ow = overlap(x1min, x1max, x2min, x2max);
        double oh = overlap(y1min, y1max, y2min, y2max);

        double intersection = ow * oh;
        double union = o1.getWidth() * o1.getHeight() + o2.getWidth() * o2.getHeight() - intersection;
        return intersection / union;
    }

    /** Performs non-maximum suppression (NMS) on objects, using their IOU with threshold to match pairs. */
    public static void nms(List<DetectedObject> objects, double iouThreshold) {
        for (int i = 0; i < objects.size(); i++) {
            for (int j = 0; j < objects.size(); j++) {
                DetectedObject o1 = objects.get(i);
                DetectedObject o2 = objects.get(j);
                if (o1 != null && o2 != null
                        && o1.getPredictedClass() == o2.getPredictedClass()
                        && o1.getConfidence() < o2.getConfidence()
                        && iou(o1, o2) > iouThreshold) {
                    objects.set(i, null);
                }
            }
        }
        Iterator<DetectedObject> it = objects.iterator();
        while (it.hasNext()) {
            if (it.next() == null) {
                it.remove();
            }
        }
    }

    /**
     * Given the network output and a detection threshold (in range 0 to 1) determine the objects detected by
     * the network.<br>
     * Supports minibatches - the returned {@link DetectedObject} instances have an example number index.<br>
     *
     * Note that the dimensions are grid cell units - for example, with 416x416 input, 32x downsampling by the network
     * (before getting to the Yolo2OutputLayer) we have 13x13 grid cells (each corresponding to 32 pixels in the input
     * image). Thus, a centerX of 5.5 would be xPixels=5.5x32 = 176 pixels from left. Widths and heights are similar:
     * in this example, a with of 13 would be the entire image (416 pixels), and a height of 6.5 would be 6.5/13 = 0.5
     * of the image (208 pixels).
     *
     * @param boundingBoxPriors as given to Yolo2OutputLayer
     * @param networkOutput 4d activations out of the network
     * @param confThreshold Detection threshold, in range 0.0 (least strict) to 1.0 (most strict). Objects are returned
     *                     where predicted confidence is >= confThreshold
     * @param nmsThreshold  passed to {@link #nms(List, double)} (0 == disabled) as the threshold for intersection over union (IOU)
     * @return List of detected objects
     */
    public static List<DetectedObject> getPredictedObjects(INDArray boundingBoxPriors, INDArray networkOutput, double confThreshold, double nmsThreshold){
        if(networkOutput.rank() != 4){
            throw new IllegalStateException("Invalid network output activations array: should be rank 4. Got array "
                    + "with shape " + Arrays.toString(networkOutput.shape()));
        }
        if(confThreshold < 0.0 || confThreshold > 1.0){
            throw new IllegalStateException("Invalid confidence threshold: must be in range [0,1]. Got: " + confThreshold);
        }

        // FIXME: int cast
        //Activations format: [mb, 5b+c, h, w]
        int mb = (int) networkOutput.size(0);
        int h = (int) networkOutput.size(2);
        int w = (int) networkOutput.size(3);
        int b = (int) boundingBoxPriors.size(0);
        int c = (int) (networkOutput.size(1)/b)-5;  //input.size(1) == b * (5 + C) -> C = (input.size(1)/b) - 5

        //Reshape from [minibatch, B*(5+C), H, W] to [minibatch, B, 5+C, H, W] to [minibatch, B, 5, H, W]
        INDArray output5 = networkOutput.dup('c').reshape(mb, b, 5+c, h, w);
        INDArray predictedConfidence = output5.get(all(), all(), point(4), all(), all());    //Shape: [mb, B, H, W]
        INDArray softmax = output5.get(all(), all(), interval(5, 5+c), all(), all());

        List<DetectedObject> out = new ArrayList<>();
        for( int i=0; i<mb; i++ ){
            for( int x=0; x<w; x++ ){
                for( int y=0; y<h; y++ ){
                    for( int box=0; box<b; box++ ){
                        double conf = predictedConfidence.getDouble(i, box, y, x);
                        if(conf < confThreshold){
                            continue;
                        }

                        double px = output5.getDouble(i, box, 0, y, x); //Originally: in 0 to 1 in grid cell
                        double py = output5.getDouble(i, box, 1, y, x); //Originally: in 0 to 1 in grid cell
                        double pw = output5.getDouble(i, box, 2, y, x); //In grid units (for example, 0 to 13)
                        double ph = output5.getDouble(i, box, 3, y, x); //In grid units (for example, 0 to 13)

                        //Convert the "position in grid cell" to "position in image (in grid cell units)"
                        px += x;
                        py += y;


                        INDArray sm;
                        try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                            sm = softmax.get(point(i), point(box), all(), point(y), point(x)).dup();
                        }
                        sm = sm.transpose();    //Convert to row vector

                        out.add(new DetectedObject(i, px, py, pw, ph, sm, conf));
                    }
                }
            }
        }

        if (nmsThreshold > 0) {
            nms(out, nmsThreshold);
        }
        return out;
    }
}
