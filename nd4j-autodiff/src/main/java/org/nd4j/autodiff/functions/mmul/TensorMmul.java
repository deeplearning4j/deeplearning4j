package org.nd4j.autodiff.functions.mmul;

import com.google.common.primitives.Ints;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.functions.AbstractBinaryReduceFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.functions.Variable;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static org.nd4j.linalg.util.ArrayUtil.convertNegativeIndices;
import static org.nd4j.linalg.util.ArrayUtil.copyOfRangeFrom;

/**
 * Created by agibsonccc on 4/14/17.
 */
public class TensorMmul<X extends Field<X>> extends AbstractBinaryReduceFunction<X> {
    private int argNum;
    private int[][] axes;
    private DifferentialFunctionFactory<X> differentialFunctionFactory;
    private boolean addedEdges = false;

    public TensorMmul(Graph<NDArrayInformation, OpState> graph,
                      DifferentialFunction<X> i_v1,
                      DifferentialFunction<X> i_v2,
                      DifferentialFunctionFactory<X> differentialFunctionFactory,
                      int[][] dimensions,int argNum) {
        super(graph, i_v1, i_v2);
        this.axes = dimensions;
        this.argNum = argNum;
        this.differentialFunctionFactory = differentialFunctionFactory;
        if(!addedEdges)
            addEdges(graph,i_v1,i_v2,functionName());
    }


    @Override
    protected void addEdges(Graph<NDArrayInformation,OpState> graph,
                            DifferentialFunction<X> i_v1,
                            DifferentialFunction<X> i_v2,
                            String opName) {
        if(i_v1.getValue() instanceof ArrayField && axes != null && !addedEdges) {
            addedEdges = true;
            ArrayField arrayField = (ArrayField) i_v1.getValue();
            ArrayField secondVal = (ArrayField) i_v2.getValue();

            addEdges(graph,i_v1,i_v2,opName,
                    OpState.OpType.ACCUMULATION,
                    ArrayUtil.getTensorMmulShape(arrayField.getInput().getShape(),
                            secondVal.getInput().getShape(),
                            axes),new Object[]{argNum});

        }

    }

    /**
     * Get the value of this function
     *
     * @return
     */
    @Override
    protected X doGetValue() {
        return differentialFunctionFactory.getMFactory().tensorMmul(larg(),rarg(),axes);
    }



    @Override
    public String functionName() {
        return "tensorMmul";
    }



    @Override
    public DifferentialFunction<X> diff(Variable<X> i_v1) {
        return doTensorMmul(argNum,larg(),rarg(),dimensions);
    }



    private DifferentialFunction<X> doTensorMmul(int argNum,
                                                 DifferentialFunction<X> a,
                                                 DifferentialFunction<X> b,
                                                 int...dimensions) {
        if (a.getValue() instanceof ArrayField) {
            ArrayField xField = (ArrayField) a.getValue();
            ArrayField yField = (ArrayField) b.getValue();
            int[] aDimensions;
            int[] bDimensions;
            if (dimensions.length == 1) {
                aDimensions = copyOfRangeFrom(
                        xField.getInput().getShape().length,
                        xField.getInput().getShape().length - dimensions[0],
                        xField.getInput().getShape().length);
                bDimensions = copyOfRangeFrom(yField.getInput().getShape().length,
                        0,
                        yField.getInput().getShape().length);
            } else {
                aDimensions = new int[0];
                bDimensions = new int[0];
            }

            if (aDimensions.length != bDimensions.length)
                throw new IllegalStateException("A and b must be the same rank");

            int[] outputShape = ArrayUtil.getTensorMmulShape(xField.getInput().getShape(),yField.getInput().getShape(),axes);
            int axesSummed = aDimensions.length;
            DifferentialFunction<X> x, y;
            int xRank, yRank;
            int[] xAxesSummed, yAxesSummed;
            int[] g_axes_from_Y;
            int gRank = 0;
            if (argNum == 0) {
                x = a;
                y = b;
                xRank = aDimensions.length;
                yRank = bDimensions.length;
                xAxesSummed = aDimensions;
                yAxesSummed = bDimensions;
                g_axes_from_Y = copyOfRangeFrom(gRank, xRank - axesSummed, gRank);

            } else if (argNum == 1) {
                x = b;
                y = a;
                xRank = bDimensions.length;
                yRank = aDimensions.length;
                xAxesSummed = bDimensions;
                yAxesSummed = aDimensions;
                g_axes_from_Y = copyOfRangeFrom(gRank, 0, yRank - axesSummed);

            } else
                throw new IllegalArgumentException("Arg num must be 0 or 1");

            xAxesSummed = convertNegativeIndices(xRank, xAxesSummed);
            yAxesSummed = convertNegativeIndices(yRank, yAxesSummed);

            int[] yAxesIgnored = ArrayUtil.removeIndex(ArrayUtil.range(0, yRank), yAxesSummed);
          //  DifferentialFunction<X> tensorDot = tensorMmul(x, new int[][]{g_axes_from_Y, yAxesIgnored});
            int[][] sortedAxesPairs = ArrayUtil.zip(xAxesSummed, yAxesSummed);
            Arrays.sort(sortedAxesPairs, new Comparator<int[]>() {
                @Override
                public int compare(int[] o1, int[] o2) {
                    return Ints.compare(o1[1], o2[1]);
                }
            });

            List<Integer> forwardPermFirst = new ArrayList<>();
            for (int i = 0; i < xRank; i++) {
                if (!Ints.contains(xAxesSummed, i))
                    forwardPermFirst.add(i);
            }

            for (int[] arr : sortedAxesPairs) {
                forwardPermFirst.add(arr[0]);
            }

            int[] forwardPerm = Ints.toArray(forwardPermFirst);
            Arrays.sort(forwardPerm);
            int[] reversePermutation = ArrayUtil.argsort(forwardPerm);
            return differentialFunctionFactory.permute(this,reversePermutation);

        }

        throw new IllegalStateException("Op type must be ArrayField");

    }
}
