package org.eclipse.deeplearning4j.ggml;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.convert.GGMLToSameDiffConverter;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLTensorInfo;
import org.nd4j.ggml.format.GGUFReader;
import org.nd4j.ggml.quantization.DequantizerFactory;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Runs the production streaming path for the real Tensor G3 token embedding in a fresh JVM.
 * Keeping this separate from allocator stress probes ensures its RSS baseline matches Android's
 * disposable importer process rather than inheriting unrelated native allocator arenas.
 */
@Slf4j
public class TensorG3StreamingDequantRssTest {

    private static final String MODEL_PATH =
            System.getProperty("user.home") + "/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf";
    private static final long MAX_STREAMING_RSS_GROWTH_MB = 1_000;

    private static long rssMb() {
        try {
            for (String line : Files.readAllLines(Paths.get("/proc/self/status"))) {
                if (line.startsWith("VmRSS:")) {
                    return Long.parseLong(line.replaceAll("\\D+", "")) / 1024;
                }
            }
        } catch (Exception ignored) {
        }
        return -1;
    }

    @Test
    public void productionStreamingTokenEmbeddingKeepsTransientRssBounded() throws Exception {
        File model = new File(MODEL_PATH);
        assertTrue(model.isFile(), "model missing: " + model);

        ConversionOptions options = ConversionOptions.builder()
                .quantizationMode(ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_MATMUL)
                .targetDataType(DataType.FLOAT)
                .embeddingDataType(DataType.HALF)
                .build();
        GGMLToSameDiffConverter converter = new GGMLToSameDiffConverter(options);
        Method streaming = GGMLToSameDiffConverter.class.getDeclaredMethod(
                "dequantizeTensorStreaming", GGUFReader.class, GGMLTensorInfo.class, DataType.class);
        streaming.setAccessible(true);

        try (GGUFReader reader = new GGUFReader(model)) {
            GGMLTensorInfo embedding = reader.getMetadata().getTensors().stream()
                    .filter(info -> "token_embd.weight".equals(info.getName()))
                    .findFirst()
                    .orElseThrow(() -> new IllegalStateException("token_embd.weight is missing"));
            assertEquals(GGMLDataType.GGML_TYPE_Q6_K, embedding.getDataType());

            try (INDArray warmup = Nd4j.scalar(0.0f)) {
                assertEquals(0.0f, warmup.getFloat(0));
            }
            long baselineMb = rssMb();
            AtomicBoolean running = new AtomicBoolean(true);
            AtomicLong peakMb = new AtomicLong(baselineMb);
            Thread sampler = new Thread(() -> {
                while (running.get()) {
                    peakMb.accumulateAndGet(rssMb(), Math::max);
                    try {
                        Thread.sleep(10);
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                }
            }, "streaming-dequant-rss-sampler");
            sampler.setDaemon(true);
            sampler.start();

            try (INDArray output = (INDArray) streaming.invoke(
                    converter, reader, embedding, DataType.HALF)) {
                assertEquals(embedding.getNumElements(), output.length());
                assertEquals(DataType.HALF, output.dataType());
                assertArrayEquals(new long[]{248_320, 1_024}, output.shape());
                int blockElements = embedding.getDataType().getBlockSize();
                int blockBytes = Math.toIntExact(
                        embedding.getDataType().calculateStorageBytes(blockElements));
                byte[] encoded = new byte[blockBytes];
                reader.readTensorDataRange(embedding, 0, encoded, 0, blockBytes);
                float[] expected = DequantizerFactory.dequantize(
                        encoded, embedding.getDataType(), blockElements);
                for (int index : new int[]{0, 1, 31, 127, 255}) {
                    assertEquals(expected[index], output.getFloat(index), 0.01f,
                            "streamed value mismatch at " + index);
                }
            } finally {
                running.set(false);
                sampler.join();
            }

            long growthMb = peakMb.get() - baselineMb;
            log.info("STREAM_DEQ_PROBE baseline_mb={} peak_mb={} growth_mb={}",
                    baselineMb, peakMb.get(), growthMb);
            assertTrue(growthMb < MAX_STREAMING_RSS_GROWTH_MB,
                    "streaming token embedding grew RSS by " + growthMb
                            + " MB; expected less than " + MAX_STREAMING_RSS_GROWTH_MB + " MB");
        }
    }
}
