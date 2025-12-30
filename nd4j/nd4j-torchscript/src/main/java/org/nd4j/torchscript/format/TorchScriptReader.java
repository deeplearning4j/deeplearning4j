/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.torchscript.format;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.torchscript.TorchScriptImportException;
import org.nd4j.torchscript.ir.PickleParser;
import org.nd4j.torchscript.ir.TorchScriptGraph;

import java.io.*;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Reader for TorchScript .pt files.
 *
 * <p>TorchScript files are ZIP archives containing:
 * <ul>
 *   <li>data.pkl: Pickled model structure and metadata</li>
 *   <li>code/: Generated Python code (optional)</li>
 *   <li>constants.pkl: Pickled constants</li>
 *   <li>data/: Binary tensor data files (0, 1, 2, ...)</li>
 * </ul>
 */
@Slf4j
public class TorchScriptReader implements Closeable {

    private final File file;
    private final ZipFile zipFile;

    private TorchScriptGraph graph;
    private Map<String, TorchScriptTensorInfo> tensorInfos;
    private Object parsedData;
    private Object parsedConstants;

    /**
     * Create a new TorchScript reader.
     *
     * @param file the TorchScript .pt file
     * @throws IOException if file cannot be opened
     */
    public TorchScriptReader(File file) throws IOException {
        this.file = file;
        this.zipFile = new ZipFile(file);
        this.tensorInfos = new LinkedHashMap<>();
    }

    /**
     * Read and parse the model graph.
     *
     * @return the parsed graph
     * @throws TorchScriptImportException if parsing fails
     */
    public TorchScriptGraph readGraph() throws TorchScriptImportException {
        if (graph != null) {
            return graph;
        }

        try {
            // Read data.pkl
            byte[] dataPkl = readZipEntry("data.pkl");
            if (dataPkl == null) {
                // Try alternate locations
                dataPkl = readZipEntry("archive/data.pkl");
            }
            if (dataPkl == null) {
                dataPkl = readZipEntry("model.pkl");
            }

            if (dataPkl != null) {
                parsedData = PickleParser.parse(dataPkl);
                graph = extractGraphFromData(parsedData);
            } else {
                log.warn("No data.pkl found in TorchScript archive");
                graph = TorchScriptGraph.builder().build();
            }

            // Read constants.pkl if present
            byte[] constantsPkl = readZipEntry("constants.pkl");
            if (constantsPkl != null) {
                parsedConstants = PickleParser.parse(constantsPkl);
            }

            return graph;

        } catch (IOException e) {
            throw new TorchScriptImportException("Failed to read TorchScript graph", e);
        }
    }

    /**
     * Read tensor info list from the archive.
     *
     * @return list of tensor information
     * @throws TorchScriptImportException if reading fails
     */
    public List<TorchScriptTensorInfo> readTensorInfos() throws TorchScriptImportException {
        if (!tensorInfos.isEmpty()) {
            return new ArrayList<>(tensorInfos.values());
        }

        try {
            // Scan for tensor data files
            Enumeration<? extends ZipEntry> entries = zipFile.entries();
            Map<Integer, Long> dataSizes = new HashMap<>();

            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                String name = entry.getName();

                // Look for data/ files (tensor data)
                if (name.startsWith("data/") && !entry.isDirectory()) {
                    String indexStr = name.substring(5); // Remove "data/"
                    try {
                        int index = Integer.parseInt(indexStr);
                        dataSizes.put(index, entry.getSize());
                    } catch (NumberFormatException e) {
                        log.debug("Skipping non-numeric data entry: {}", name);
                    }
                }
                // Also check archive/data/
                else if (name.startsWith("archive/data/") && !entry.isDirectory()) {
                    String indexStr = name.substring(13);
                    try {
                        int index = Integer.parseInt(indexStr);
                        dataSizes.put(index, entry.getSize());
                    } catch (NumberFormatException e) {
                        log.debug("Skipping non-numeric data entry: {}", name);
                    }
                }
            }

            // If we have parsed data, try to extract tensor info from it
            if (parsedData != null) {
                extractTensorInfoFromData(parsedData, dataSizes);
            } else {
                // Read data.pkl to get tensor info
                readGraph();
                if (parsedData != null) {
                    extractTensorInfoFromData(parsedData, dataSizes);
                }
            }

            log.info("Found {} tensors in TorchScript archive", tensorInfos.size());
            return new ArrayList<>(tensorInfos.values());

        } catch (Exception e) {
            throw new TorchScriptImportException("Failed to read tensor infos", e);
        }
    }

    /**
     * Read tensor data for a specific tensor.
     *
     * @param tensorInfo the tensor to read
     * @return raw tensor data bytes
     * @throws IOException if reading fails
     */
    public byte[] readTensorData(TorchScriptTensorInfo tensorInfo) throws IOException {
        String dataPath = "data/" + tensorInfo.getDataOffset();
        byte[] data = readZipEntry(dataPath);

        if (data == null) {
            // Try archive/data/
            dataPath = "archive/data/" + tensorInfo.getDataOffset();
            data = readZipEntry(dataPath);
        }

        if (data == null) {
            throw new IOException("Tensor data not found: " + tensorInfo.getName());
        }

        return data;
    }

    /**
     * Get metadata for the model.
     *
     * @return model metadata
     * @throws TorchScriptImportException if reading fails
     */
    public TorchScriptMetadata getMetadata() throws TorchScriptImportException {
        TorchScriptGraph g = readGraph();
        List<TorchScriptTensorInfo> tensors = readTensorInfos();

        return TorchScriptMetadata.builder()
                .format(TorchScriptFormat.TORCHSCRIPT_PT)
                .graph(g)
                .tensors(tensors)
                .build();
    }

    /**
     * Read a ZIP entry as bytes.
     *
     * @param entryName the entry name
     * @return the bytes or null if not found
     * @throws IOException if reading fails
     */
    private byte[] readZipEntry(String entryName) throws IOException {
        ZipEntry entry = zipFile.getEntry(entryName);
        if (entry == null) {
            return null;
        }

        try (InputStream is = zipFile.getInputStream(entry)) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int len;
            while ((len = is.read(buffer)) != -1) {
                baos.write(buffer, 0, len);
            }
            return baos.toByteArray();
        }
    }

    /**
     * Extract graph structure from parsed pickle data.
     */
    private TorchScriptGraph extractGraphFromData(Object data) {
        TorchScriptGraph.TorchScriptGraphBuilder builder = TorchScriptGraph.builder();

        if (data instanceof Map) {
            Map<?, ?> map = (Map<?, ?>) data;
            // Look for model structure
            log.debug("Parsed data keys: {}", map.keySet());
        } else if (data instanceof List) {
            // Some formats use lists
            log.debug("Parsed data is a list with {} elements", ((List<?>) data).size());
        } else if (data instanceof PickleParser.ReduceResult) {
            // Most common case - a reduced object
            PickleParser.ReduceResult result = (PickleParser.ReduceResult) data;
            log.debug("Parsed data is ReduceResult: {}", result.callable);
            extractFromReduceResult(builder, result);
        }

        return builder.build();
    }

    /**
     * Extract graph info from a ReduceResult.
     */
    @SuppressWarnings("unchecked")
    private void extractFromReduceResult(TorchScriptGraph.TorchScriptGraphBuilder builder,
                                         PickleParser.ReduceResult result) {
        if (result.getState() instanceof Map) {
            Map<String, Object> state = (Map<String, Object>) result.getState();
            // Look for _modules, _parameters, _buffers
            if (state.containsKey("_modules")) {
                Object modules = state.get("_modules");
                log.debug("Found _modules in state");
            }
            if (state.containsKey("_parameters")) {
                Object params = state.get("_parameters");
                log.debug("Found _parameters in state");
            }
        }
    }

    /**
     * Extract tensor info from parsed pickle data.
     */
    @SuppressWarnings("unchecked")
    private void extractTensorInfoFromData(Object data, Map<Integer, Long> dataSizes) {
        if (data instanceof PickleParser.ReduceResult) {
            PickleParser.ReduceResult result = (PickleParser.ReduceResult) data;
            extractTensorsFromReduceResult(result, "", dataSizes, new HashSet<>());
        } else if (data instanceof Map) {
            extractTensorsFromMap((Map<?, ?>) data, "", dataSizes, new HashSet<>());
        }
    }

    @SuppressWarnings("unchecked")
    private void extractTensorsFromReduceResult(PickleParser.ReduceResult result, String prefix,
                                                 Map<Integer, Long> dataSizes, Set<Object> visited) {
        if (visited.contains(result)) return;
        visited.add(result);

        if (result.getState() instanceof Map) {
            extractTensorsFromMap((Map<?, ?>) result.getState(), prefix, dataSizes, visited);
        }
        if (result.args instanceof List) {
            for (Object arg : (List<?>) result.args) {
                if (arg instanceof PickleParser.ReduceResult) {
                    extractTensorsFromReduceResult((PickleParser.ReduceResult) arg, prefix, dataSizes, visited);
                }
            }
        }
    }

    @SuppressWarnings("unchecked")
    private void extractTensorsFromMap(Map<?, ?> map, String prefix,
                                        Map<Integer, Long> dataSizes, Set<Object> visited) {
        if (visited.contains(map)) return;
        visited.add(map);

        // Check for _parameters and _buffers
        if (map.containsKey("_parameters")) {
            Map<?, ?> params = (Map<?, ?>) map.get("_parameters");
            if (params != null) {
                for (Map.Entry<?, ?> entry : params.entrySet()) {
                    String name = prefix + entry.getKey();
                    extractTensorFromValue(name, entry.getValue(), dataSizes, true, false);
                }
            }
        }

        if (map.containsKey("_buffers")) {
            Map<?, ?> buffers = (Map<?, ?>) map.get("_buffers");
            if (buffers != null) {
                for (Map.Entry<?, ?> entry : buffers.entrySet()) {
                    String name = prefix + entry.getKey();
                    extractTensorFromValue(name, entry.getValue(), dataSizes, false, true);
                }
            }
        }

        // Recurse into _modules
        if (map.containsKey("_modules")) {
            Map<?, ?> modules = (Map<?, ?>) map.get("_modules");
            if (modules != null) {
                for (Map.Entry<?, ?> entry : modules.entrySet()) {
                    String newPrefix = prefix.isEmpty() ? entry.getKey() + "." : prefix + entry.getKey() + ".";
                    Object module = entry.getValue();
                    if (module instanceof PickleParser.ReduceResult) {
                        extractTensorsFromReduceResult((PickleParser.ReduceResult) module, newPrefix, dataSizes, visited);
                    } else if (module instanceof Map) {
                        extractTensorsFromMap((Map<?, ?>) module, newPrefix, dataSizes, visited);
                    }
                }
            }
        }
    }

    @SuppressWarnings("unchecked")
    private void extractTensorFromValue(String name, Object value, Map<Integer, Long> dataSizes,
                                         boolean isParameter, boolean isBuffer) {
        if (value == null) return;

        // Handle PersistentId which points to tensor data
        if (value instanceof PickleParser.PersistentId) {
            PickleParser.PersistentId pid = (PickleParser.PersistentId) value;
            // The id typically contains (storage_type, key, device, size)
            if (pid.id instanceof List) {
                List<?> idList = (List<?>) pid.id;
                if (idList.size() >= 4) {
                    // idList[1] is typically the storage key (index)
                    Object keyObj = idList.get(1);
                    int storageIndex = -1;
                    if (keyObj instanceof Number) {
                        storageIndex = ((Number) keyObj).intValue();
                    } else if (keyObj instanceof String) {
                        try {
                            storageIndex = Integer.parseInt((String) keyObj);
                        } catch (NumberFormatException e) {
                            // Not a numeric key
                        }
                    }

                    if (storageIndex >= 0 && dataSizes.containsKey(storageIndex)) {
                        long dataSize = dataSizes.get(storageIndex);

                        TorchScriptTensorInfo info = TorchScriptTensorInfo.builder()
                                .name(name)
                                .dataOffset(storageIndex) // Store index as offset
                                .dataSize(dataSize)
                                .requiresGrad(isParameter)
                                .isBuffer(isBuffer)
                                .dataType(TorchScriptDataType.FLOAT32) // Default, may need to infer
                                .build();

                        tensorInfos.put(name, info);
                        log.debug("Found tensor: {} (storage index: {}, size: {})", name, storageIndex, dataSize);
                    }
                }
            }
        } else if (value instanceof PickleParser.ReduceResult) {
            // Might be a tensor wrapped in a reduce result
            PickleParser.ReduceResult rr = (PickleParser.ReduceResult) value;
            if (rr.args instanceof List) {
                for (Object arg : (List<?>) rr.args) {
                    if (arg instanceof PickleParser.PersistentId) {
                        extractTensorFromValue(name, arg, dataSizes, isParameter, isBuffer);
                    }
                }
            }
        }
    }

    /**
     * Get the file being read.
     *
     * @return the file
     */
    public File getFile() {
        return file;
    }

    @Override
    public void close() throws IOException {
        if (zipFile != null) {
            zipFile.close();
        }
    }
}
