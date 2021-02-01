/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.imports.listeners;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

@Slf4j
public class ImportDebugListener extends BaseListener {

    @Override
    public boolean isActive(Operation operation) {
        return true;
    }

    public enum OnFailure {EXCEPTION, LOG};

    private File baseDir;
    private FilenameFunction function;
    private boolean checkShapesOnly;
    private double fpEps;
    private OnFailure onFailure;
    private boolean logPass;

    public ImportDebugListener(Builder b){
        this.baseDir = b.baseDir;
        this.function = b.function;
        this.checkShapesOnly = b.checkShapesOnly;
        this.fpEps = b.fpEps;
        this.onFailure = b.onFailure;
        this.logPass = b.logPass;
    }

    @Override
    public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {
        //No op

        for( int i=0; i<outputs.length; i++ ) {
            File f = function.getFileFor(baseDir, op.getName(), i);
            if(!f.exists()){
                log.warn("Skipping check for op {} output {}, no file found: {}", op.getName(), i, f.getAbsolutePath());
                continue;
            }

            INDArray arr;
            try {
                arr = Nd4j.createFromNpyFile(f);
            } catch (Throwable t){
                throw new RuntimeException("Error loading numpy file for op " + op.getName() + " - " + f.getAbsolutePath(), t);
            }

            if(arr.dataType() != outputs[i].dataType()){
                String msg = "Datatype does not match: " + op.getName() + ", output " + i + " - TF=" +
                        arr.dataType() + ", SD=" + outputs[i].dataType() + "; TF shape info: " + arr.shapeInfoToString() +
                        " vs. SD shape info: " + outputs[i].shapeInfoToString();
                switch (onFailure){
                    case EXCEPTION:
                        throw new RuntimeException(msg);
                    case LOG:
                        log.error(msg);
                        break;
                    default:
                        throw new RuntimeException();
                }
            }

            if(arr.isEmpty()){
                if(!outputs[i].isEmpty()){
                    String msg = "TF array is empty but SameDiff output " + i + " is not. TF shape info: " + arr.shapeInfoToString() +
                            " vs. SD shape info: " + outputs[i].shapeInfoToString();
                    switch (onFailure){
                        case EXCEPTION:
                            throw new RuntimeException(msg);
                        case LOG:
                            log.error(msg);
                            break;
                        default:
                            throw new RuntimeException();
                    }
                }
            }

            if(!arr.equalShapes(outputs[i])){
                String msg = "SameDiff output " + i + " does not match TF shape. TF shape info: " + arr.shapeInfoToString() +
                        " vs. SD shape info: " + outputs[i].shapeInfoToString();
                switch (onFailure){
                    case EXCEPTION:
                        throw new RuntimeException(msg);
                    case LOG:
                        log.error(msg);
                        break;
                    default:
                        throw new RuntimeException();
                }
            }

            if(checkShapesOnly){
                continue;
            }

            boolean eq = arr.equalsWithEps(outputs[i], fpEps);
            if(!eq){
                String msg = "SameDiff output " + i + " does not match TF values with eps=" + fpEps + ". TF shape info: " + arr.shapeInfoToString() +
                        " vs. SD shape info: " + outputs[i].shapeInfoToString();
                switch (onFailure){
                    case EXCEPTION:
                        throw new RuntimeException(msg);
                    case LOG:
                        log.error(msg);
                        break;
                    default:
                        throw new RuntimeException();
                }
            }

            if(logPass){
                log.info("Passed: {} output {}", op.getName(), i);
            }
        }
    }

    public static Builder builder(File rootDir){
        return new Builder(rootDir);
    }


    public static class Builder {

        private File baseDir;
        private FilenameFunction function = new DefaultFilenameFunction();
        private boolean checkShapesOnly = false;
        private double fpEps = 1e-5;
        private OnFailure onFailure = OnFailure.EXCEPTION;
        private boolean logPass = false;


        public Builder(@NonNull File baseDir){
            this.baseDir = baseDir;
        }

        public Builder onFailure(OnFailure onFailure){
            this.onFailure = onFailure;
            return this;
        }

        public Builder filenameFunction(@NonNull FilenameFunction fn){
            this.function = fn;
            return this;
        }

        public Builder checkShapesOnly(boolean shapesOnly){
            this.checkShapesOnly = shapesOnly;
            return this;
        }

        public Builder floatingPointEps(double eps){
            this.fpEps = eps;
            return this;
        }

        public Builder logPass(boolean logPass){
            this.logPass = logPass;
            return this;
        }

        public ImportDebugListener build(){
            return new ImportDebugListener(this);
        }

    }

    public interface FilenameFunction {
        File getFileFor(File rootDir, String opName, int outputNum);

    }

    public static class DefaultFilenameFunction implements FilenameFunction {

        @Override
        public File getFileFor(File rootDir, String opName, int outputNum){
            return new File(rootDir, opName + "__" + outputNum + ".npy");
        }
    }
}
