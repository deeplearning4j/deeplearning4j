
/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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


package org.datavec.python;


import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Emulates multiples interpreters in a single interpreter.
 * This works by simply obfuscating/de-obfuscating variable names
 * such that only the required subset of the global namespace is "visible"
 * at any given time.
 * By default, there exists a "main" context emulating the default interpreter
 * and cannot be deleted.
 * @author Fariz Rahman
 */


public class PythonContextManager {

    private static Set<String> contexts = new HashSet<>();
    private static AtomicBoolean init = new AtomicBoolean(false);
    private static String currentContext;
    private static final String MAIN_CONTEXT = "main";
    static {
        init();
    }

    private static void init() {
        if (init.get()) return;
        new PythonExecutioner();
        init.set(true);
        currentContext = MAIN_CONTEXT;
        contexts.add(currentContext);
    }


    public static void addContext(String contextName) throws PythonException {
        if (!validateContextName(contextName)) {
            throw new PythonException("Invalid context name: " + contextName);
        }
        contexts.add(contextName);
    }

    public static boolean hasContext(String contextName) {
        return contexts.contains(contextName);
    }


    public static boolean validateContextName(String s) {
        if (s.length() == 0) return false;
        if (!Character.isJavaIdentifierStart(s.charAt(0))) return false;
        for (int i = 1; i < s.length(); i++)
            if (!Character.isJavaIdentifierPart(s.charAt(i)))
                return false;
        return true;
    }

    private static String getContextPrefix(String contextName) {
        return "__collapsed__" + contextName + "__";
    }

    private static String getCollapsedVarNameForContext(String varName, String contextName) {
        return getContextPrefix(contextName) + varName;
    }

    private static String expandCollapsedVarName(String varName, String contextName) {
        String prefix = "__collapsed__" + contextName + "__";
        return varName.substring(prefix.length());

    }

    private static void collapseContext(String contextName) {
        PythonObject globals = Python.globals();
        PythonObject keysList = Python.list(globals.attr("keys").call());
        int numKeys = Python.len(keysList).toInt();
        for (int i = 0; i < numKeys; i++) {
            PythonObject key = keysList.get(i);
            String keyStr = key.toString();
            if (!((keyStr.startsWith("__") && keyStr.endsWith("__")) || keyStr.startsWith("__collapsed_"))) {
                String collapsedKey = getCollapsedVarNameForContext(keyStr, contextName);
                PythonObject val = globals.attr("pop").call(key);
                globals.set(new PythonObject(collapsedKey), val);
            }
        }
    }

    private static void expandContext(String contextName) {
        String prefix = getContextPrefix(contextName);
        PythonObject globals = Python.globals();
        PythonObject keysList = Python.list(globals.attr("keys").call());
        int numKeys = Python.len(keysList).toInt();
        for (int i = 0; i < numKeys; i++) {
            PythonObject key = keysList.get(i);
            String keyStr = key.toString();
            if (keyStr.startsWith(prefix)) {
                String expandedKey = expandCollapsedVarName(keyStr, contextName);
                PythonObject val = globals.attr("pop").call(key);
                globals.set(new PythonObject(expandedKey), val);
            }
        }

    }

    public static void setContext(String contextName) throws PythonException{
        if (contextName.equals(currentContext)) {
            return;
        }
        if (!hasContext(contextName)) {
            addContext(contextName);
        }
        collapseContext(currentContext);
        expandContext(contextName);
        currentContext = contextName;

    }

    public static void setMainContext() {
        try{
            setContext(MAIN_CONTEXT);
        }
        catch (PythonException pe){
            throw new RuntimeException(pe);
        }

    }

    public static String getCurrentContext() {
        return currentContext;
    }

    public static void deleteContext(String contextName) throws PythonException {
        if (contextName.equals(MAIN_CONTEXT)) {
            throw new PythonException("Can not delete main context!");
        }
        if (contextName.equals(currentContext)) {
            throw new PythonException("Can not delete current context!");
        }
        String prefix = getContextPrefix(contextName);
        PythonObject globals = Python.globals();
        PythonObject keysList = Python.list(globals.attr("keys").call());
        int numKeys = Python.len(keysList).toInt();
        for (int i = 0; i < numKeys; i++) {
            PythonObject key = keysList.get(i);
            String keyStr = key.toString();
            if (keyStr.startsWith(prefix)) {
                globals.attr("__delitem__").call(key);
            }
        }
        contexts.remove(contextName);
    }

    public static void deleteNonMainContexts() {
        try{
            setContext(MAIN_CONTEXT); // will never fail
        for (String c : contexts.toArray(new String[0])) {
            if (!c.equals(MAIN_CONTEXT)) {
                deleteContext(c); // will never fail
            }
        }
        }catch(Exception e){
            throw new RuntimeException(e);
        }
    }

    public String[] getContexts() {
        return contexts.toArray(new String[0]);
    }

}
