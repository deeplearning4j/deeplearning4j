package org.deeplearning4j.plot;

import com.google.common.base.Function;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.Condition;
import org.nd4j.linalg.learning.AdaGrad;
import static org.nd4j.linalg.ops.transforms.Transforms.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;


import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.nd4j.linalg.ops.transforms.Transforms.sign;

/**
 * Tsne calculation
 * @author Adam Gibson
 */
public class Tsne {

    private int maxIter = 1000;
    private float realMin = 1e-12f;
    private float initialMomentum = 0.5f;
    private float finalMomentum = 0.8f;
    private int eta = 500;
    private final float minGain = 1e-2f;
    private float momentum = initialMomentum;
    private int switchMomentumIteration = 100;
    private boolean normalize = true;
    private boolean usePca = false;
    private int stopLyingIteration = 250;
    private float tolerance = 1e-5f;
    private float learningRate = 1e-1f;
    private AdaGrad adaGrad;
    private boolean useAdaGrad = true;
    private float perplexity = 30f;

    private String commandTemplate = "python /tmp/tsne.py --path %s --ndims %d --perplexity %.3f --initialdims %s --labels %s";



    private static ClassPathResource r = new ClassPathResource("/scripts/tsne.py");
    private static ClassPathResource r2 = new ClassPathResource("/scripts/render.py");


    static {
        loadIntoTmp();
    }

    private static void loadIntoTmp() {

        File script = new File("/tmp/tsne.py");


        try {
            List<String> lines = IOUtils.readLines(r.getInputStream());
            FileUtils.writeLines(script, lines);

        } catch (IOException e) {
            throw new IllegalStateException("Unable to load python file");

        }


        File script2 = new File("/tmp/render.py");


        try {
            List<String> lines2 = IOUtils.readLines(r2.getInputStream());
            FileUtils.writeLines(script2, lines2);

        } catch (IOException e) {
            throw new IllegalStateException("Unable to load python file");

        }

    }
    private static Logger log = LoggerFactory.getLogger(Tsne.class);

    public Tsne(
            int maxIter,
            float realMin,
            float initialMomentum,
            float finalMomentum,
            int eta,
            float momentum,
            int switchMomentumIteration,
            boolean normalize,
            boolean usePca,
            int stopLyingIteration,
            float tolerance,float learningRate,boolean useAdaGrad,float perplexity) {
        this.tolerance = tolerance;
        this.useAdaGrad = useAdaGrad;
        this.learningRate = learningRate;
        this.stopLyingIteration = stopLyingIteration;
        this.maxIter = maxIter;
        this.realMin = realMin;
        this.normalize = normalize;
        this.initialMomentum = initialMomentum;
        this.usePca = usePca;
        this.finalMomentum = finalMomentum;
        this.eta = eta;
        this.momentum = momentum;
        this.switchMomentumIteration = switchMomentumIteration;
        this.perplexity = perplexity;
    }

    /**
     * Computes a gaussian kernel
     * given a vector of squared euclidean distances
     *
     * @param d
     * @param beta
     * @return
     */
    public Pair<INDArray,INDArray> hBeta(INDArray d,float beta) {
        INDArray P =  exp(d.neg().muli(beta));
        INDArray sum = P.sum(0);

        INDArray H = log(sum).addi(beta).muli(d.mul(P).sum(0)).divi(sum);


        return new Pair<>(H,P);
    }




    /**
     * Convert data to probability
     * co-occurrences
     * @param d the data to convert
     * @param u the perplexity of the model
     * @param tolerance the tolerance for convergence
     * @return the probabilities of co-occurrence
     */
    public INDArray d2p(final INDArray d,final float u,final float tolerance) {
        int n = d.rows();
        final INDArray p = Nd4j.create(n, n);
        final INDArray beta = Nd4j.ones(n, 1);
        final float logU = (float) Math.log(u);
        log.info("Calculating probabilities of data similarities..");
        ExecutorService service = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        for(int i = 0; i < n; i++) {
            if(i % 500 == 0)
                log.info("Handled " + i + " records");
            final int j = i;
            service.submit(new Runnable() {
                @Override
                public void run() {
                    float betaMin = Float.NEGATIVE_INFINITY;
                    float betaMax = Float.POSITIVE_INFINITY;

                    Pair<INDArray,INDArray> pair =  hBeta(d.getRow(j),beta.get(j));

                    INDArray hDiff = pair.getFirst().sub(logU);
                    int tries = 0;

                    Condition c = new Condition() {
                        @Override
                        public Boolean apply(Number input) {
                            return input.floatValue() > tolerance;
                        }

                        @Override
                        public Boolean apply(IComplexNumber input) {
                            return input.absoluteValue().floatValue() > tolerance;
                        }
                    };




                    Condition c2 = new Condition() {
                        @Override
                        public Boolean apply(Number input) {
                            return input.floatValue() > 0;
                        }

                        @Override
                        public Boolean apply(IComplexNumber input) {
                            return input.absoluteValue().floatValue() > 0;
                        }
                    };

                    while(BooleanIndexing.and(abs(hDiff),c) && tries < 50) {
                        if(BooleanIndexing.and(hDiff,c2)) {
                            betaMin = beta.get(j);
                            if(Float.isInfinite(betaMax))
                                beta.putScalar(j,beta.get(j) * 2);
                            else
                                beta.putScalar(j,(beta.get(j) + betaMax) / 2);
                        }
                        else {
                            betaMax = beta.get(j);
                            if(Float.isInfinite(betaMin))
                                beta.putScalar(j,beta.get(j) / 2);
                            else
                                beta.putScalar(j,(beta.get(j) + betaMin) / 2);
                        }

                        pair = hBeta(d.getRow(j),beta.get(j));
                        hDiff = pair.getFirst().sub(logU);
                        tries++;
                    }

                    p.putRow(j,pair.getSecond());

                }
            });
        }

        service.shutdown();
        try {
            service.awaitTermination(1, TimeUnit.DAYS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }


        return p;

    }






    /**
     *
     * @param X
     * @param nDims
     * @param perplexity
     */
    public  INDArray calculate(INDArray X,int nDims,float perplexity) {
        if(usePca)
            X = PCA.pca(X, Math.min(50,X.columns()),normalize);
            //normalization (don't normalize again after pca)
        else if(normalize) {
            X = X.sub(X.min(Integer.MAX_VALUE));
            X = X.divi(X.max(Integer.MAX_VALUE));
            X = X.subiRowVector(X.mean(0));
        }
        if(nDims > X.columns())
            nDims = X.columns();




        INDArray sumX =  pow(X, 2).sum(1);
        INDArray D = X.mmul(
                X.transpose())
                .muli(-2)
                .addiColumnVector(
                        sumX.transpose())
                .addiRowVector(sumX);




        INDArray y = Nd4j.randn(X.rows(),nDims).muli(1e-3f);
        INDArray yIncs = Nd4j.create(y.shape());
        INDArray gains = Nd4j.ones(y.shape());

        INDArray p = d2p(D,perplexity,tolerance);

        //set 0 along the diagonal
        Nd4j.doAlongDiagonal(p,new Function<Number, Number>() {
            @Override
            public Number apply(Number input) {
                return 0;
            }
        });

        p = p.add(p.transpose()).muli(0.5f);
        p = max(p.div(p.sum(Integer.MAX_VALUE)),realMin);
        BooleanIndexing.applyWhere(p,new Condition() {
            @Override
            public Boolean apply(Number input) {
                return Float.isNaN(input.floatValue());
            }

            @Override
            public Boolean apply(IComplexNumber input) {
                return Float.isNaN(input.realComponent().floatValue());
            }
        },new Function<Number, Number>() {
            @Override
            public Number apply(Number input) {
                return realMin;
            }
        });

        float constant = Nd4j.getBlasWrapper().dot(p, log(p));
        //lie for better local minima
        p.muli(4);
        float epsilon = 500;
        float costCheck = Float.NEGATIVE_INFINITY;
        for(int i = 0; i < maxIter; i++) {

            INDArray sumY =  pow(y, 2).sum(1);
            //Student-t distribution
            INDArray num = y.mmul(
                    y.transpose())
                    .muli(-2)
                    .addiRowVector(sumY)
                    .addiColumnVector(sumY.transpose())
                    .addi(1).rdivi(1);
            Nd4j.doAlongDiagonal(num,new Function<Number, Number>() {
                @Override
                public Number apply(Number input) {
                    return 0;
                }
            });


            // normalize to get probabilities
            INDArray  q =  max(num.divi(num.sum(Integer.MAX_VALUE).addi(1e-6f)), realMin);
            INDArray L = p.sub(q).muli(num);
            INDArray yGrads = Nd4j.diag(L.sum(0)).subi(L).muli(4).mmul(y);
            if(useAdaGrad) {
                if(adaGrad == null) {
                    adaGrad = new AdaGrad(yGrads.shape());
                    adaGrad.setMasterStepSize(learningRate);
                }
                yGrads.muli(adaGrad.getLearningRates(yGrads));
            }
            else
                yGrads.muli(learningRate);

            gains = max(gains.add(.2f).muli
                    (sign(yGrads).eps(sign(yIncs)))
                    .addi(gains.mul(0.8f))
                    .muli(sign(yGrads).eq(sign(yIncs))),minGain);

            yIncs = yIncs.mul(momentum).subi(gains.mul(yGrads).muli(epsilon));
            y.addi(yIncs).subiRowVector(y.mean(0));

            if(i == switchMomentumIteration)
                momentum = finalMomentum;

            INDArray logQ = log(q);
            float dot = Nd4j.getBlasWrapper().dot(p, logQ);
            float cost = constant - dot;
            if(!Float.isInfinite(costCheck)) {
                float diff = Math.abs(costCheck - cost);
                if(diff <= 1e-6 && i > stopLyingIteration)
                    break;
            }

            costCheck = cost;
            log.info("Cost " + cost + " at iteration " + i);


            if(i == stopLyingIteration)
                p.divi(4);

        }

        return y;
    }



    /**
     * Plot tsne
     * @param matrix the matrix to plot
     * @param nDims the number
     * @param initialDims
     * @param labels
     * @throws IOException
     */
    public void plot(INDArray matrix,int nDims,int initialDims,List<String> labels) throws IOException {

        INDArray y = calculate(matrix,nDims,perplexity);


        String path = writeMatrix(y);
        String labelPath = UUID.randomUUID().toString();

        File f = new File(labelPath);
        FileUtils.writeLines(f,labels);
        String command = String.format(commandTemplate,path,nDims,perplexity,initialDims,labelPath);
        Process is = Runtime.getRuntime().exec(command);

        log.info("Std out " + IOUtils.readLines(is.getInputStream()).toString());
        log.error(IOUtils.readLines(is.getErrorStream()).toString());

    }




    protected String writeMatrix(INDArray matrix) throws IOException {
        String filePath = System.getProperty("java.io.tmpdir") + File.separator +  UUID.randomUUID().toString();
        File write = new File(filePath);
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
        write.deleteOnExit();
        for(int i = 0; i < matrix.rows(); i++) {
            INDArray row = matrix.getRow(i);
            StringBuffer sb = new StringBuffer();
            for(int j = 0; j < row.length(); j++) {
                sb.append(String.format("%.10f", row.get(j)));
                if(j < row.length() - 1)
                    sb.append(",");
            }
            sb.append("\n");
            String line = sb.toString();
            bos.write(line.getBytes());
            bos.flush();
        }

        bos.close();
        return filePath;
    }

    public static class Builder {
        private int maxIter = 1000;
        private float realMin = 1e-6f;
        private float initialMomentum = 5e-1f;
        private float finalMomentum = 8e-1f;
        private int eta = 500;
        private float momentum = 5e-1f;
        private int switchMomentumIteration = 100;
        private boolean normalize = true;
        private boolean usePca = false;
        private int stopLyingIteration = 100;
        private float tolerance = 1e-5f;
        private float learningRate = 1e-1f;
        private boolean useAdaGrad = true;
        private float perplexity = 30;



        public Builder perplexity(float perplexity) {
            this.perplexity = perplexity;
            return this;
        }

        public Builder useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }

        public Builder learningRate(float learningRate) {
            this.learningRate = learningRate;
            return this;
        }


        public Builder tolerance(float tolerance) {
            this.tolerance = tolerance;
            return this;
        }

        public Builder stopLyingIteration(int stopLyingIteration) {
            this.stopLyingIteration = stopLyingIteration;
            return this;
        }

        public Builder usePca(boolean usePca) {
            this.usePca = usePca;
            return this;
        }

        public Builder normalize(boolean normalize) {
            this.normalize = normalize;
            return this;
        }

        public Builder setMaxIter(int maxIter) {
            this.maxIter = maxIter;
            return this;
        }

        public Builder setRealMin(float realMin) {
            this.realMin = realMin;
            return this;
        }

        public Builder setInitialMomentum(float initialMomentum) {
            this.initialMomentum = initialMomentum;
            return this;
        }

        public Builder setFinalMomentum(float finalMomentum) {
            this.finalMomentum = finalMomentum;
            return this;
        }

        public Builder setEta(int eta) {
            this.eta = eta;
            return this;
        }

        public Builder setMomentum(float momentum) {
            this.momentum = momentum;
            return this;
        }

        public Builder setSwitchMomentumIteration(int switchMomentumIteration) {
            this.switchMomentumIteration = switchMomentumIteration;
            return this;
        }

        public Tsne build() {
            return new Tsne(maxIter, realMin, initialMomentum, finalMomentum, eta, momentum, switchMomentumIteration,normalize,usePca,stopLyingIteration,tolerance,learningRate,useAdaGrad,perplexity);
        }

    }
}
