package org.nd4j.linalg.api.indexing.shape;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.Indices;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class IndexShapeTests extends BaseNd4jTest {

    public IndexShapeTests(Nd4jBackend backend) {
        super(backend);
    }



    private  int[] shape = {1,1,2,1,3,4,5,1};

    @Test
    public void testSinglePoint() {
        /*
        Assumes all indexes are filled out.
        Test simple general point case
         */
        int[] assertion = {2, 1, 4, 5, 1};
        INDArrayIndex[] indexes = new INDArrayIndex[] {
                NDArrayIndex.point(0),
                NDArrayIndex.point(0),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.point(0),
                NDArrayIndex.all()
        };

        int[] testShape = Indices.shape(shape,indexes);
        assertArrayEquals(assertion,testShape);

        int[] secondAssertion = {1, 2, 1, 5, 1};
        INDArrayIndex[] otherCase = new INDArrayIndex[] {
                NDArrayIndex.all(),
                NDArrayIndex.point(0),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.point(0),
                NDArrayIndex.point(0)

        };

        assertArrayEquals(secondAssertion,Indices.shape(shape,otherCase));


        int[] thridAssertion = {1, 2, 1, 4, 5, 1};
        INDArrayIndex[] thirdCase = new INDArrayIndex[] {
                NDArrayIndex.all(),
                NDArrayIndex.point(0),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.point(0),

        };
        assertArrayEquals(thridAssertion,Indices.shape(shape,thirdCase));

    }

    @Test
    public void testInterval() {
        int[] basicAssertion = {1, 1, 1, 1, 3, 1, 2, 1};
        INDArrayIndex[] basicTest = {
                NDArrayIndex.all()
                ,NDArrayIndex.all(),
                NDArrayIndex.interval(0,1),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.interval(1,2),
                NDArrayIndex.interval(2,4),
                NDArrayIndex.all()
        };
        assertArrayEquals(basicAssertion,Indices.shape(shape,basicTest));

    }


    @Test
    public void testNewAxis() {
        //normal prepend
        int[] prependAssertion = {1, 1, 1, 1, 2, 1, 3, 4, 5, 1};
        INDArrayIndex[] prependTest = {
                NDArrayIndex.newAxis(),
                NDArrayIndex.newAxis(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),

        };

        assertArrayEquals(prependAssertion,Indices.shape(shape,prependTest));

        //test setting for particular indexes.
        //when an all is encountered before a new axis,
        //it is assumed that new axis must occur at the destination
        //where the new axis was specified
        int[] addToMiddle = {1, 1, 2, 1, 1, 1, 3, 4, 5, 1};
        INDArrayIndex[] setInMiddleTest = {
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.newAxis(),
                NDArrayIndex.newAxis(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
        };
        assertArrayEquals(addToMiddle,Indices.shape(shape,setInMiddleTest));

        //test prepending AND adding to middle
        int[] prependAndAddToMiddleAssertion = {1,1,1, 1, 2, 1, 1, 1, 3, 4, 5, 1};

        INDArrayIndex[] prependAndMiddle = {
                NDArrayIndex.newAxis(),
                NDArrayIndex.newAxis(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.newAxis(),
                NDArrayIndex.newAxis(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
        };
        assertArrayEquals(prependAndAddToMiddleAssertion,Indices.shape(shape,prependAndMiddle));

    }


    @Override
    public char ordering() {
        return 'f';
    }
}
