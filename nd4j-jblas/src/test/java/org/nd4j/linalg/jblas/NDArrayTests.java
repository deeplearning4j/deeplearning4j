package org.nd4j.linalg.jblas;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import static org.junit.Assert.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;


/**
 * NDArrayTests
 * @author Adam Gibson
 */
public class NDArrayTests extends org.nd4j.linalg.api.test.NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(NDArrayTests.class);


    @Test
    public void testMatrixVector() {
        double[][] data = new double[][]{
                {1,2,3,4},
                {5,6,7,8}
        };


        Nd4j.factory().setOrder('f');
        double[] mmul = {1,2,3,4};

        DoubleMatrix d = new DoubleMatrix(data);
        INDArray d2 = Nd4j.create(data);
        assertEquals(d.rows,d2.rows());
        assertEquals(d.columns,d2.columns());
        verifyElements(d,d2);

        INDArray toMmulD2 = Nd4j.create(mmul).transpose();
        DoubleMatrix toMmulD = new DoubleMatrix(mmul);


        assertEquals(d.rows,d2.rows());
        assertEquals(d.columns,d2.columns());

        assertEquals(toMmulD.rows,toMmulD2.rows());
        assertEquals(toMmulD.columns,toMmulD2.columns());

        DoubleMatrix mmulResultD = d.mmul(toMmulD);
        INDArray mmulResultD2 = d2.mmul(toMmulD2);

        verifyElements(mmulResultD,mmulResultD2);





        Nd4j.factory().setOrder('c');


    }

    @Test
    public void testTransposeCompat() {
        Nd4j.factory().setOrder('f');
        DoubleMatrix dReshaped = DoubleMatrix.linspace(1,8,8).reshape(2,4);
        INDArray nReshaped = Nd4j.linspace(1, 8, 8).reshape(2,4);
        verifyElements(dReshaped,nReshaped);
        DoubleMatrix d = dReshaped.transpose();
        INDArray n = nReshaped.transpose();
        assertTrue(Arrays.equals(d.data,ArrayUtil.doubleCopyOf(n.data())));

        verifyElements(d,n);

        double[] data = {-0.005221025552600622,-0.004393004812300205,-0.0037941287737339735,0.010821177624166012,-0.014784608036279678,0.004485765006393194,-0.0050201937556266785,0.00406963936984539,-0.0035935400519520044,-2.6935606729239225E-4,-0.008975246921181679,-0.0058696032501757145,0.014686450362205505,-0.0022672158665955067,0.012981601990759373,0.00354340230114758,-0.006773303262889385,2.3264715855475515E-4,-0.018955592066049576,0.0022182068787515163,-0.022435732185840607,0.003009446896612644,-0.001288946601562202,0.0011316317832097411,0.012024265713989735,-0.014933056198060513,0.006626023445278406,-0.004271078854799271,0.007069796789437532,-0.014591346494853497,0.004245881922543049,-0.00903915986418724,0.002792439656332135,0.007792876102030277,-9.262353414669633E-4,-7.420211331918836E-4,0.013156365603208542,-0.009934891015291214,-0.018974831327795982,0.008624176494777203,0.006072960793972015,0.002770041348412633,-0.02252178080379963,-0.007812395226210356,0.007836085744202137,-0.00615832069888711,-0.011055439710617065,7.374464767053723E-4,0.021642161533236504,4.191290936432779E-4,0.0015816434752196074,-0.02199694700539112,-0.0015862988075241446,-0.010778022930026054,-0.005442730151116848,-0.008786597289144993,-0.006276786793023348,-0.005609111860394478,-0.012948143295943737,-0.004546381067484617,0.015863541513681412,0.002912799594923854,-0.0030169105157256126,-0.004328157752752304,0.00953906960785389,-0.0034996727481484413,-0.020346807315945625,0.003978618886321783,0.0019423117628321052,-0.006504336837679148,-5.052743945270777E-4,-0.006164539139717817,3.504526102915406E-4,4.8193871043622494E-4,0.013964755460619926,0.01079760491847992,0.008938292041420937,-0.00979834794998169,0.005000569857656956,0.012097761034965515,-0.006645648740231991,-0.006269180215895176,-0.005769916344434023,-0.004605017602443695,0.005792100913822651,0.012450706213712692,0.00538474228233099,0.011030850000679493,-0.003173709148541093,-0.004666883498430252,0.014975965023040771,-0.008902465924620628,-0.0048470874316990376,0.006161658093333244,0.006186444777995348,-7.273021037690341E-4,0.0030987728387117386,0.004152507055550814,0.006515987683087587,-0.0010259306291118264,-0.002500643488019705,-0.00624958798289299,-0.0025901934131979942,-0.01363024115562439,0.011106634512543678,-0.008492453023791313,0.012225041165947914,0.010147170163691044,0.005233681760728359,-0.007156160660088062,-0.003972422331571579,-0.005760961212217808,0.0014825743855908513,0.02239028364419937,-0.0072554475627839565,-0.01063522882759571,-0.005555273033678532,0.017090942710638046,-0.006399329751729965,0.009354383684694767,0.0012814776273444295,-0.010278275236487389,0.019066374748945236,0.019803497940301895,0.010189397260546684,-0.012875289656221867,0.004658989608287811,-0.014810028485953808,0.006855623330920935,-0.012391027994453907,0.012881937436759472,-0.01865105889737606,0.00103187735658139,-0.0026612081564962864,-0.00474441098049283,0.008176152594387531,0.009247633628547192,-0.001050183898769319,-0.002326537389308214,0.0019891788251698017,-0.01805080473423004,-0.0031765911262482405,-0.010865583084523678,0.0013048049295321107,0.001743039465509355,-0.010462394915521145,0.006381515879184008,-0.010733230970799923,-0.015365495346486568,0.007117144763469696,0.00863807462155819,0.0076784477569162846,-0.01110463310033083,-0.004248939920216799,-0.00169592653401196,-0.0075967758893966675,0.0014429340371862054,-0.017970828339457512,-0.015638219192624092,0.007891322486102581,0.00620066886767745,-0.010048598051071167,-5.066935555078089E-4,-0.016040397807955742,-0.0012570320395752788,-0.0024861590936779976,-0.0020170388743281364,0.017319003120064735,0.010896338149905205,0.007520229090005159,0.0067520709708333015,0.006276045460253954,-0.01143174059689045,-6.349519826471806E-4,0.014258163049817085,-0.001049129175953567,0.01700351946055889,0.013120148330926895,0.002119696233421564,1.8147178343497217E-4,0.004363402258604765,-0.01674243062734604,-0.006698529236018658,0.019545456394553185,0.01102487277239561,-0.008940811268985271,-0.005405755713582039,0.010684389621019363,0.0077524976804852486,-0.024562828242778778,-0.00419618608430028,0.003311727661639452,0.00977502390742302,5.088065518066287E-4,0.0015424853190779686,0.005078576970845461,0.013699470087885857,0.0026570989284664392,0.0018757919315248728,-0.003375320229679346,-0.013585430569946766,-0.011017655953764915,-0.002602857304736972,0.01102753821760416,-0.009821008890867233,0.008224218152463436,0.0036263500805944204,0.004874897189438343,-0.0045783426612615585,8.395070326514542E-4,0.008432632312178612,0.0015309917507693172,0.004779332783073187,-0.010448360815644264,-0.001519222161732614,-0.004239152185618877,4.746414924738929E-5,-0.010563353076577187,0.00946486834436655,-0.020225441083312035,-0.013003045693039894,0.01034512184560299,-0.009619565680623055,-0.009377307258546352,0.0020278533920645714,0.00163548206910491,-0.0022132443264126778,0.009237022139132023,-0.004036027938127518,-0.0012084163026884198,-0.008177138864994049,0.009920339100062847,0.006989401765167713,0.01277521625161171,0.014595700427889824,-0.010948852635920048,2.5860303139779717E-5,-0.012776975519955158,0.0033487414475530386,-0.005722139496356249,-0.01208622008562088,-0.010535686276853085,0.007737908512353897,-0.004314272664487362,0.003475845092907548,-0.020825747400522232,-0.015113264322280884,0.022749634459614754,-0.0023766090162098408,-0.006387890782207251};
        float[] copy = ArrayUtil.floatCopyOf(data);
        DoubleMatrix d3 = new DoubleMatrix(25,10,data);
        INDArray other = Nd4j.create(copy,new int[]{25,10});
        verifyElements(d3,other);
        verifyElements(d3.transpose(),other.transpose());
        assertTrue(Arrays.equals(d3.transpose().data,ArrayUtil.doubleCopyOf(other.transpose().data())));
        double[] toMMul = {0.003841338213533163,0.0,0.01873396337032318,0.0,0.0,0.028475480154156685,0.0,0.02141464874148369,0.0,0.02895500883460045,0.03538861125707626,0.0,0.01675591431558132,0.0,0.04823039844632149,0.0,0.008317765779793262,0.0,0.0388733372092247,0.0};
        DoubleMatrix dToMmul = new DoubleMatrix(2,10,toMMul);
        DoubleMatrix result = dToMmul.mmul(d3.transpose());
        float[] toMmul2 = ArrayUtil.floatCopyOf(toMMul);
        INDArray toMMulF = Nd4j.create(toMmul2,new int[]{2,10});



        INDArray result2 = toMMulF.mmul(other.transpose());
        verifyElements(result,result2);


        Nd4j.factory().setOrder('c');






    }


    @Test
    public void testFortranRavel() {
        double[][] data = new double[][] {
                {1,2,3,4},
                {5,6,7,8}
        };

        INDArray toRavel = Nd4j.create(data);
        Nd4j.factory().setOrder('f');
        INDArray toRavelF = Nd4j.create(data);
        INDArray ravel = toRavel.ravel();
        INDArray ravelF = toRavelF.ravel();
        assertEquals(ravel,ravelF);
        Nd4j.factory().setOrder('c');

    }


    @Test
    public void testNorm1() {
        DoubleMatrix norm1 = DoubleMatrix.linspace(1,8,8).reshape(2,4);
        INDArray norm1NDArray = Nd4j.linspace(1, 8, 8).reshape(2,4);
        assertEquals(norm1.norm1(),norm1NDArray.norm1(Integer.MAX_VALUE).get(0),1e-1);
    }



    @Test
    public void testFortranReshapeMatrix() {
        double[][] data = new double[][]{
                {1,2,3,4},
                {5,6,7,8}
        };

        Nd4j.factory().setOrder('f');

        DoubleMatrix d = new DoubleMatrix(data);
        INDArray d2 = Nd4j.create(data);
        assertEquals(d.rows, d2.rows());
        assertEquals(d.columns, d2.columns());
        verifyElements(d, d2);


        DoubleMatrix reshapedD = d.reshape(4,2);
        INDArray reshapedD2 = d2.reshape(4,2);
        verifyElements(reshapedD,reshapedD2);
        Nd4j.factory().setOrder('c');


    }






    @Test
    public void testFortranCreation() {
        double[][] data = new double[][]{
                {1,2,3,4},
                {5,6,7,8}
        };


        Nd4j.factory().setOrder('f');
        float[][] mmul = {{1,2,3,4},{5,6,7,8}};

        INDArray d2 = Nd4j.create(data);
        verifyElements(mmul,d2);
    }


    @Test
    public void testMatrixMatrix() {
        double[][] data = new double[][]{
                {1, 2, 3, 4},
                {5, 6, 7, 8}
        };


        Nd4j.factory().setOrder('f');
        double[][] mmul = {{1, 2, 3, 4}, {5, 6, 7, 8}};

        DoubleMatrix d = new DoubleMatrix(data).reshape(4, 2);
        INDArray d2 = Nd4j.create(data).reshape(4, 2);
        assertEquals(d.rows, d2.rows());
        assertEquals(d.columns, d2.columns());
        verifyElements(d, d2);

        INDArray toMmulD2 = Nd4j.create(mmul);
        DoubleMatrix toMmulD = new DoubleMatrix(mmul);

        DoubleMatrix mmulResultD = d.mmul(toMmulD);
        INDArray mmulResultD2 = d2.mmul(toMmulD2);
        verifyElements(mmulResultD, mmulResultD2);


        Nd4j.factory().setOrder('c');
    }

    @Test
    public void testVectorVector() {
        DoubleMatrix d = new DoubleMatrix(2,1);
        d.data = new double[]{1,2};
        DoubleMatrix d2 = new DoubleMatrix(1,2);
        d2.data = new double[]{3,4};

        INDArray d3 = Nd4j.create(new double[]{1, 2}).reshape(2,1);
        INDArray d4 = Nd4j.create(new double[]{3, 4});

        assertEquals(d.rows,d3.rows());
        assertEquals(d.columns,d3.columns());

        assertEquals(d2.rows,d4.rows());
        assertEquals(d2.columns,d4.columns());

        DoubleMatrix resultMatrix = d.mmul(d2);



        INDArray resultNDArray = d3.mmul(d4);
        verifyElements(resultMatrix,resultNDArray);

    }


    @Test
    public void testVector() {
        Nd4j.factory().setOrder('f');

        DoubleMatrix dJblas = DoubleMatrix.linspace(1,4,4);
        INDArray d = Nd4j.linspace(1, 4, 4);
        verifyElements(dJblas,d);
        Nd4j.factory().setOrder('c');


    }
    @Test
    public void testRowVectorOps() {
        if(Nd4j.factory().order() ==  NDArrayFactory.C) {
            INDArray twoByTwo = Nd4j.create(new float[]{1, 3, 2, 4}, new int[]{2, 2});
            INDArray toAdd = Nd4j.create(new float[]{1, 2}, new int[]{2});
            twoByTwo.addiRowVector(toAdd);
            INDArray assertion = Nd4j.create(new float[]{2, 5,3, 6}, new int[]{2, 2});
            assertEquals(assertion,twoByTwo);

        }



    }

    @Test
    public void testColumnVectorOps() {
        if(Nd4j.factory().order() == NDArrayFactory.C) {
            INDArray twoByTwo = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
            INDArray toAdd = Nd4j.create(new float[]{1, 2}, new int[]{2, 1});
            twoByTwo.addiColumnVector(toAdd);
            INDArray assertion = Nd4j.create(new float[]{2, 3, 5, 6}, new int[]{2, 2});
            assertEquals(assertion,twoByTwo);


        }


    }

    @Test
    public void testReshapeCompatibility() {
        Nd4j.factory().setOrder('f');
        DoubleMatrix oneThroughFourJblas = DoubleMatrix.linspace(1,4,4).reshape(2,2);
        DoubleMatrix fiveThroughEightJblas = DoubleMatrix.linspace(5,8,4).reshape(2,2);
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4).reshape(2,2);
        INDArray fiveThroughEight = Nd4j.linspace(5, 8, 4).reshape(2,2);
        verifyElements(oneThroughFourJblas,oneThroughFour);
        verifyElements(fiveThroughEightJblas,fiveThroughEight);
        Nd4j.factory().setOrder('c');

    }

    @Test
    public void testRowSumCompat() {
        Nd4j.factory().setOrder('f');
        DoubleMatrix rowsJblas = DoubleMatrix.linspace(1,8,8).reshape(2,4);
        INDArray rows = Nd4j.linspace(1, 8, 8).reshape(2,4);
        verifyElements(rowsJblas,rows);

        INDArray rowSums = rows.sum(1);
        DoubleMatrix jblasRowSums = rowsJblas.rowSums();
        verifyElements(jblasRowSums,rowSums);


        float[][] data = new float[][]{
                {1,2},{3,4}
        };

        INDArray rowSumsData = Nd4j.create(data);
        Nd4j.factory().setOrder('c');
        INDArray rowSumsCOrder = Nd4j.create(data);
        assertEquals(rowSumsData,rowSumsCOrder);
        INDArray rowSumsDataSum = rowSumsData.sum(1);
        INDArray rowSumsCOrderSum = rowSumsCOrder.sum(1);
        assertEquals(rowSumsDataSum,rowSumsCOrderSum);
        INDArray assertion = Nd4j.create(new float[]{3, 7});
        assertEquals(assertion,rowSumsCOrderSum);
        assertEquals(assertion,rowSumsDataSum);
    }



    protected void verifyElements(float[][] d,INDArray d2) {
        for(int i = 0; i < d2.rows(); i++) {
            for(int j = 0; j < d2.columns(); j++) {
                float test1 =  d[i][j];
                float test2 = d2.get(i,j);
                assertEquals(test1,test2,1e-6);
            }
        }
    }


    protected void verifyElements(DoubleMatrix d,INDArray d2) {
        if(d.isVector() && d2.isVector())
            for(int j = 0; j < d2.length(); j++) {
                float test1 = (float) d.get(j);
                float test2 =  d2.get(j);
                assertEquals(test1,test2,1e-6);
            }

        else {
            for(int i = 0; i < d.rows; i++) {
                for(int j = 0; j < d.columns; j++) {
                    float test1 = (float) d.get(i,j);
                    float test2 = d2.get(i,j);
                    assertEquals(test1,test2,1e-6);
                }
            }
        }

    }

}
