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


package org.nd4j.python4j;

import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.Loader;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class PythonProcess {
    private static String pythonExecutable = Loader.load(org.bytedeco.cpython.python.class);
    public static String runAndReturn(String... arguments)throws IOException, InterruptedException{
        String[] allArgs = new String[arguments.length + 1];
        for (int i = 0; i < arguments.length; i++){
            allArgs[i + 1] = arguments[i];
        }
        allArgs[0] = pythonExecutable;
        ProcessBuilder pb = new ProcessBuilder(allArgs);
        Process process = pb.start();
        String out = IOUtils.toString(process.getInputStream(), StandardCharsets.UTF_8);
        process.waitFor();
        return out;

    }

    public static void run(String... arguments)throws IOException, InterruptedException{
        String[] allArgs = new String[arguments.length + 1];
        for (int i = 0; i < arguments.length; i++){
            allArgs[i + 1] = arguments[i];
        }
        allArgs[0] = pythonExecutable;
        ProcessBuilder pb = new ProcessBuilder(allArgs);
        pb.inheritIO().start().waitFor();
    }
    public static void pipInstall(String packageName) throws PythonException{
        try{
            run("-m", "pip", "install", packageName);
        }catch(Exception e){
            throw new PythonException("Error installing package " + packageName, e);
        }

    }

    public static void pipInstall(String packageName, String version){
        pipInstall(packageName + "==" + version);
    }

    public static void pipUninstall(String packageName) throws PythonException{
        try{
            run("-m", "pip", "uninstall", packageName);
        }catch(Exception e){
            throw new PythonException("Error uninstalling package " + packageName, e);
        }

    }
    public static void pipInstallFromGit(String gitRepoUrl){
        if (!gitRepoUrl.contains("://")){
            gitRepoUrl = "git://" + gitRepoUrl;
        }
        try{
            run("-m", "pip", "install", "git+", gitRepoUrl);
        }catch(Exception e){
            throw new PythonException("Error installing package from " + gitRepoUrl, e);
        }

    }

    public static String getPackageVersion(String packageName){
        String out;
        try{
            out = runAndReturn("-m", "pip", "show", packageName);
        } catch (Exception e){
            throw new PythonException("Error finding version for package " + packageName, e);
        }

        if (!out.contains("Version: ")){
            throw new PythonException("Can't find package " + packageName);
        }
        String pkgVersion  = out.split("Version: ")[1].split(System.lineSeparator())[0];
        return pkgVersion;
    }

    public static boolean isPackageInstalled(String packageName){
        try{
            String out = runAndReturn("-m", "pip", "show", packageName);
            return !out.isEmpty();
        }catch (Exception e){
            throw new PythonException("Error checking if package is installed: " +packageName, e);
        }

    }

    public static void pipInstallFromRequirementsTxt(String path){
        try{
            run("-m", "pip", "install","-r", path);
        }catch (Exception e){
            throw new PythonException("Error installing packages from " + path, e);
        }
    }

    public static void pipInstallFromSetupScript(String path, boolean inplace){

        try{
            run(path, inplace?"develop":"install");
        }catch (Exception e){
            throw new PythonException("Error installing package from " + path, e);
        }

    }

}