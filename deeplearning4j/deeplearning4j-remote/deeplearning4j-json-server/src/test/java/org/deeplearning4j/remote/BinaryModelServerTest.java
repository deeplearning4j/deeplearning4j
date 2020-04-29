package org.deeplearning4j.remote;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.remote.helpers.ImageConversionUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.After;
import org.junit.Test;
import org.nd4j.adapters.InferenceAdapter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.remote.clients.JsonRemoteInference;
import org.nd4j.remote.clients.serde.BinaryDeserializer;
import org.nd4j.remote.clients.serde.BinarySerializer;
import org.nd4j.remote.clients.serde.impl.IntegerSerde;
import org.nd4j.common.resources.Resources;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.deeplearning4j.parallelism.inference.InferenceMode.SEQUENTIAL;
import static org.junit.Assert.*;

@Slf4j
public class BinaryModelServerTest extends BaseDL4JTest {
    private final int PORT = 18080;

    @After
    public void pause() throws Exception {
        // TODO: the same port was used in previous test and not accessible immediately. Might be better solution.
        TimeUnit.SECONDS.sleep(2);
    }

    // Internal test for locally defined serializers
    @Test
    public void testBufferedImageSerde() {
            BinarySerializer<BufferedImage> serde = new BinaryModelServerTest.BufferedImageSerde();
            BufferedImage image = ImageConversionUtils.makeRandomBufferedImage(28,28,1);
            byte[] serialized = serde.serialize(image);

            BufferedImage deserialized = ((BufferedImageSerde) serde).deserialize(serialized);
            int originalSize = image.getData().getDataBuffer().getSize();
            assertEquals(originalSize, deserialized.getData().getDataBuffer().getSize());
            for (int i = 0; i < originalSize; ++i) {
                assertEquals(deserialized.getData().getDataBuffer().getElem(i),
                             image.getData().getDataBuffer().getElem(i));
            }
    }

    @Test
    public void testImageToINDArray() {
        INDArray data = ImageConversionUtils.makeRandomImageAsINDArray(28,28,1);
        assertNotNull(data);
    }

    @Test
    public void testMlnMnist_ImageInput() throws Exception {

        val modelFile = Resources.asFile("models/mnist/mnist-model.zip");
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(modelFile);

        val server = new JsonModelServer.Builder<BufferedImage, Integer>(net)
                .outputSerializer(new IntegerSerde())
                .inputBinaryDeserializer(new BufferedImageSerde())
                .inferenceAdapter(new InferenceAdapter<BufferedImage, Integer>() {
                    @Override
                    public MultiDataSet apply(BufferedImage input) {
                        INDArray data = null;
                        try {
                            data = new Java2DNativeImageLoader().asMatrix(input);
                            data = data.reshape(1, 784);
                        }
                        catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                        return new MultiDataSet(data, null);
                    }

                    @Override
                    public Integer apply(INDArray... nnOutput) {
                        return nnOutput[0].argMax().getInt(0);
                    }
                })
                .port(PORT)
                .inferenceMode(SEQUENTIAL)
                .numWorkers(1)
                .parallelMode(false)
                .build();

        val client = JsonRemoteInference.<BufferedImage, Integer>builder()
                .endpointAddress("http://localhost:" + PORT + "/v1/serving")
                .inputBinarySerializer(new BufferedImageSerde())
                .outputDeserializer(new IntegerSerde())
                .build();

        try {
            server.start();
            BufferedImage image = ImageConversionUtils.makeRandomBufferedImage(28,28,1);
            Integer result = client.predict(image);
            assertNotNull(result);

            File file = new ClassPathResource("datavec-local/imagetest/0/b.bmp").getFile();
            image = ImageIO.read(new FileInputStream(file));
            result = client.predict(image);
            assertEquals(new Integer(0), result);

            file = new ClassPathResource("datavec-local/imagetest/1/a.bmp").getFile();
            image = ImageIO.read(new FileInputStream(file));
            result = client.predict(image);
            assertEquals(new Integer(1), result);

        } catch (Exception e){
            log.error("",e);
            throw e;
        } finally {
            server.stop();
        }
    }

    @Test
    public void testMlnMnist_ImageInput_Async() throws Exception {

        val modelFile = Resources.asFile("models/mnist/mnist-model.zip");
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(modelFile);

        val server = new JsonModelServer.Builder<BufferedImage, Integer>(net)
                .outputSerializer(new IntegerSerde())
                .inputBinaryDeserializer(new BufferedImageSerde())
                .inferenceAdapter(new InferenceAdapter<BufferedImage, Integer>() {
                    @Override
                    public MultiDataSet apply(BufferedImage input) {
                        INDArray data = null;
                        try {
                            data = new Java2DNativeImageLoader().asMatrix(input);
                            data = data.reshape(1, 784);
                        }
                        catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                        return new MultiDataSet(data, null);
                    }

                    @Override
                    public Integer apply(INDArray... nnOutput) {
                        return nnOutput[0].argMax().getInt(0);
                    }
                })
                .port(PORT)
                .inferenceMode(SEQUENTIAL)
                .numWorkers(1)
                .parallelMode(false)
                .build();

        val client = JsonRemoteInference.<BufferedImage, Integer>builder()
                .endpointAddress("http://localhost:" + PORT + "/v1/serving")
                .inputBinarySerializer(new BufferedImageSerde())
                .outputDeserializer(new IntegerSerde())
                .build();

        try {
            server.start();
            BufferedImage[] images = new BufferedImage[3];
            images[0] = ImageConversionUtils.makeRandomBufferedImage(28,28,1);

            File file = new ClassPathResource("datavec-local/imagetest/0/b.bmp").getFile();
            images[1] = ImageIO.read(new FileInputStream(file));

            file = new ClassPathResource("datavec-local/imagetest/1/a.bmp").getFile();
            images[2] = ImageIO.read(new FileInputStream(file));

            Future<Integer>[] results = new Future[3];
            for (int i = 0; i < images.length; ++i) {
                results[i] = client.predictAsync(images[i]);
                assertNotNull(results[i]);
            }

            assertNotNull(results[0].get());
            assertEquals(new Integer(0), results[1].get());
            assertEquals(new Integer(1), results[2].get());

        } catch (Exception e){
            log.error("",e);
            throw e;
        } finally {
            server.stop();
        }
    }

    @Test
    public void testBinaryIn_BinaryOut() throws Exception {

        val modelFile = Resources.asFile("models/mnist/mnist-model.zip");
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(modelFile);

        val server = new JsonModelServer.Builder<BufferedImage, BufferedImage>(net)
                .outputBinarySerializer(new BufferedImageSerde())
                .inputBinaryDeserializer(new BufferedImageSerde())
                .inferenceAdapter(new InferenceAdapter<BufferedImage, BufferedImage>() {
                    @Override
                    public MultiDataSet apply(BufferedImage input) {
                        INDArray data = null;
                        try {
                            data = new Java2DNativeImageLoader().asMatrix(input);
                        }
                        catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                        return new MultiDataSet(data, null);
                    }

                    @Override
                    public BufferedImage apply(INDArray... nnOutput) {
                        return ImageConversionUtils.makeRandomBufferedImage(28,28,3);
                    }
                })
                .port(PORT)
                .inferenceMode(SEQUENTIAL)
                .numWorkers(1)
                .parallelMode(false)
                .build();

        val client = JsonRemoteInference.<BufferedImage, BufferedImage>builder()
                .endpointAddress("http://localhost:" + PORT + "/v1/serving")
                .inputBinarySerializer(new BufferedImageSerde())
                .outputBinaryDeserializer(new BufferedImageSerde())
                .build();

        try {
            server.start();
            BufferedImage image = ImageConversionUtils.makeRandomBufferedImage(28,28,1);
            BufferedImage result = client.predict(image);
            assertNotNull(result);
            assertEquals(28, result.getHeight());
            assertEquals(28, result.getWidth());

        } catch (Exception e){
            log.error("",e);
            throw e;
        } finally {
            server.stop();
        }
    }

    private static class BufferedImageSerde implements BinarySerializer<BufferedImage>, BinaryDeserializer<BufferedImage> {

        @Override
        public BufferedImage deserialize(byte[] buffer) {
            try {
                BufferedImage img = ImageIO.read(new ByteArrayInputStream(buffer));
                return img;
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }

        @Override
        public byte[] serialize(BufferedImage image) {
            try{
                val baos = new ByteArrayOutputStream();
                ImageIO.write(image, "bmp", baos);
                byte[] bytes = baos.toByteArray();
                return bytes;
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }
    }
}
