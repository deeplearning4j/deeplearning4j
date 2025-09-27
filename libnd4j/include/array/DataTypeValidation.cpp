/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Implementation of robust data type validation and error reporting
//

#include <array/DataType.h>
#include <array/DataTypeValidation.h>
#include <exceptions/allocation_exception.h>
#include <system/op_enums.h>

#include <sstream>

#include "system/op_boilerplate.h"
#include <helpers/logger.h>
namespace sd {
    
    // Static member definitions
    std::unordered_set<DataType> DataTypeValidation::compiledTypes_;
    std::unordered_map<DataType, std::string> DataTypeValidation::typeNames_;
    std::unordered_map<int, DataType> DataTypeValidation::enumMapping_;
    bool DataTypeValidation::initialized_ = false;
    
    void DataTypeValidation::initialize() {
        if (initialized_) return;
        
        populateTypeNames();
        populateCompiledTypes();
        
        // Build enum ID to DataType mapping
        for (const auto& pair : typeNames_) {
            enumMapping_[static_cast<int>(pair.first)] = pair.first;
        }
        
        initialized_ = true;
    }
    
    void DataTypeValidation::initializeIfNeeded() {
        if (!initialized_) {
            initialize();
        }
    }
    
    void DataTypeValidation::populateTypeNames() {
        if (initialized_) return;

        // All possible data types defined in DataType enum
        typeNames_[DataType::INHERIT] = "INHERIT";
        typeNames_[DataType::BOOL] = "BOOL";
        typeNames_[DataType::FLOAT8] = "FLOAT8";
        typeNames_[DataType::HALF] = "HALF";
        typeNames_[DataType::HALF2] = "HALF2";
        typeNames_[DataType::FLOAT32] = "FLOAT32";
        typeNames_[DataType::DOUBLE] = "DOUBLE";
        typeNames_[DataType::INT8] = "INT8";
        typeNames_[DataType::INT16] = "INT16";
        typeNames_[DataType::INT32] = "INT32";
        typeNames_[DataType::INT64] = "INT64";
        typeNames_[DataType::UINT8] = "UINT8";
        typeNames_[DataType::UINT16] = "UINT16";
        typeNames_[DataType::UINT32] = "UINT32";
        typeNames_[DataType::UINT64] = "UINT64";
        typeNames_[DataType::QINT8] = "QINT8";
        typeNames_[DataType::QINT16] = "QINT16";
        typeNames_[DataType::BFLOAT16] = "BFLOAT16";
        typeNames_[DataType::UTF8] = "UTF8";
        typeNames_[DataType::UTF16] = "UTF16";
        typeNames_[DataType::UTF32] = "UTF32";
        typeNames_[DataType::UNKNOWN] = "UNKNOWN";
    }
    void DataTypeValidation::populateCompiledTypes() {
        if (initialized_) return;

        // Auto-register types based on compile-time flags
        // This mirrors the logic from the CMake type system

        // Boolean type
#if defined(HAS_BOOL)
        registerCompiledType(DataType::BOOL, "BOOL");
#endif

        // Floating point types - check both CMake-generated and alias defines
#if defined(HAS_FLOAT) || defined(HAS_FLOAT32)
        registerCompiledType(DataType::FLOAT32, "FLOAT32");
#endif

#if defined(HAS_DOUBLE) || defined(HAS_FLOAT64)
        registerCompiledType(DataType::DOUBLE, "DOUBLE");
#endif

#if defined(HAS_FLOAT16) || defined(HAS_HALF)
        registerCompiledType(DataType::HALF, "HALF");
#endif

#if defined(HAS_BFLOAT16) || defined(HAS_BFLOAT)
        registerCompiledType(DataType::BFLOAT16, "BFLOAT16");
#endif

        // Integer types - check both normalized (_T) and alias forms
#if defined(HAS_INT8_T) || defined(HAS_INT8)
        registerCompiledType(DataType::INT8, "INT8");
#endif

#if defined(HAS_INT16_T) || defined(HAS_INT16)
        registerCompiledType(DataType::INT16, "INT16");
#endif

#if defined(HAS_INT32_T) || defined(HAS_INT32) || defined(HAS_INT)
        registerCompiledType(DataType::INT32, "INT32");
#endif

#if defined(HAS_INT64_T) || defined(HAS_INT64) || defined(HAS_LONG)
        registerCompiledType(DataType::INT64, "INT64");
#endif

        // Unsigned integer types
#if defined(HAS_UINT8_T) || defined(HAS_UINT8)
        registerCompiledType(DataType::UINT8, "UINT8");
#endif

#if defined(HAS_UINT16_T) || defined(HAS_UINT16)
        registerCompiledType(DataType::UINT16, "UINT16");
#endif

#if defined(HAS_UINT32_T) || defined(HAS_UINT32)
        registerCompiledType(DataType::UINT32, "UINT32");
#endif

#if defined(HAS_UINT64_T) || defined(HAS_UINT64) || defined(HAS_UNSIGNEDLONG)
        registerCompiledType(DataType::UINT64, "UINT64");
#endif

        // String types - only if string operations are enabled
#if defined(SD_ENABLE_STRING_OPERATIONS)
#if defined(HAS_STD_STRING) || defined(HAS_UTF8)
        registerCompiledType(DataType::UTF8, "UTF8");
#endif

#if defined(HAS_STD_U16STRING) || defined(HAS_UTF16)
        registerCompiledType(DataType::UTF16, "UTF16");
#endif

#if defined(HAS_STD_U32STRING) || defined(HAS_UTF32)
        registerCompiledType(DataType::UTF32, "UTF32");
#endif
#endif

    }
    
    void DataTypeValidation::registerCompiledType(DataType type, const std::string& name) {
        compiledTypes_.insert(type);
        typeNames_[type] = name;
    }
    
    bool DataTypeValidation::isValidDataType(DataType dataType) {
        initializeIfNeeded();
        return typeNames_.find(dataType) != typeNames_.end();
    }
    
    bool DataTypeValidation::isValidDataType(int dataTypeId) {
        initializeIfNeeded();
        return enumMapping_.find(dataTypeId) != enumMapping_.end();
    }

    std::vector<DataType> DataTypeValidation::getAvailableDataTypes() {
        initializeIfNeeded();
        std::vector<DataType> result;
        result.reserve(compiledTypes_.size());
        for (const auto& type : compiledTypes_) {
            result.push_back(type);
        }
        return result;
    }
    
    bool DataTypeValidation::isCompiledDataType(DataType dataType) {
        initializeIfNeeded();
        return compiledTypes_.find(dataType) != compiledTypes_.end();
    }
    
    std::string DataTypeValidation::getDataTypeErrorMessage(int dataTypeId, const char* context) {
        initializeIfNeeded();

        std::ostringstream oss;

        if (!isValidDataType(dataTypeId)) {
            oss << "Invalid data type ID: " << dataTypeId << " (not a valid enum value)";
        } else {
            DataType dt = enumMapping_[dataTypeId];
            const char* typeName = getDataTypeName(dt);

            if (!isCompiledDataType(dt)) {
              oss << "Data type " << typeName << " (ID: " << dataTypeId
                  << ") is not available in this build";
            } else {
              oss << "Unexpected error with data type " << typeName
                  << " (ID: " << dataTypeId << ") - type is valid and compiled";
            }
        }

        if (context) {
            oss << "\nContext: " << context;
        }

        // Add build configuration info with better diagnosis
#if defined(SD_SELECTIVE_TYPES)
        oss << "\nBuild configuration: Selective types enabled";

        // List the define flags that would enable missing types
        if (isValidDataType(dataTypeId)) {
            DataType dt = enumMapping_[dataTypeId];
            if (!isCompiledDataType(dt)) {
              const char* typeName = getDataTypeName(dt);
              oss << "\nTo enable this type, rebuild with the type included in SD_TYPES_LIST";
              oss << "\nExample: cmake -DSD_TYPES_LIST=\"float32;int32;int64;" << typeName << "\" ..";
            }
        }
#else
        oss << "\nBuild configuration: All types enabled (SD_SELECTIVE_TYPES not defined)";
        oss << "\nThis error suggests a build system configuration issue.";
#endif

        // Add available types
        auto availableTypes = getAvailableDataTypes();
        if (!availableTypes.empty()) {
            oss << "\nAvailable types in this build (" << availableTypes.size() << "): ";
            for (size_t i = 0; i < availableTypes.size(); ++i) {
              if (i > 0) oss << ", ";
              oss << getDataTypeName(availableTypes[i]);
            }
        } else {
            oss << "\nCRITICAL: No data types are available in this build!";
            oss << "\nThis indicates a serious build configuration problem.";

            // Debug information to help diagnose the issue
            oss << "\nDiagnostic information:";
            oss << "\n  Total possible types: " << typeNames_.size();
            oss << "\n  Compiled types: " << compiledTypes_.size();

            // Show which defines are actually set
            oss << "\n  Active defines:";
#if defined(HAS_BOOL)
            oss << " HAS_BOOL";
#endif
#if defined(HAS_FLOAT)
            oss << " HAS_FLOAT";
#endif
#if defined(HAS_FLOAT32)
            oss << " HAS_FLOAT32";
#endif
#if defined(HAS_DOUBLE)
            oss << " HAS_DOUBLE";
#endif
#if defined(HAS_INT32_T)
            oss << " HAS_INT32_T";
#endif
#if defined(HAS_INT32)
            oss << " HAS_INT32";
#endif
#if defined(HAS_INT)
            oss << " HAS_INT";
#endif
#if defined(HAS_INT64_T)
            oss << " HAS_INT64_T";
#endif
#if defined(HAS_LONG)
            oss << " HAS_LONG";
#endif
#if defined(SD_SELECTIVE_TYPES)
            oss << " SD_SELECTIVE_TYPES";
#endif

            if (compiledTypes_.empty()) {
              oss << "\n  No type defines matched - check CMake type generation";
            }
        }

        // Add suggestions based on the situation
        if (!isValidDataType(dataTypeId)) {
            oss << "\nSuggestion: Check NDArray creation and data type initialization";
        } else if (availableTypes.empty()) {
            oss << "\nSuggestion: Check CMake build configuration and type define generation";
            oss << "\nVerify that setup_type_definitions() was called in CMake";
        } else {
            oss << "\nSuggestion: Use one of the available types or rebuild with the required type enabled";
        }

        return oss.str();
    }
    
    const char* DataTypeValidation::getDataTypeName(DataType dataType) {
        initializeIfNeeded();
        auto it = typeNames_.find(dataType);
        return (it != typeNames_.end()) ? it->second.c_str() : "UNKNOWN";
    }
    
    std::string DataTypeValidation::getBuildInfo() {
        initializeIfNeeded();
        std::ostringstream oss;

        oss << "=== DATA TYPE VALIDATION BUILD INFO ===\n";

#if defined(SD_SELECTIVE_TYPES)
        oss << "Build type: Selective types (SD_SELECTIVE_TYPES defined)\n";
#else
        oss << "Build type: All types (SD_SELECTIVE_TYPES not defined)\n";
#endif

        oss << "Total possible types: " << typeNames_.size() << "\n";
        oss << "Compiled types: " << compiledTypes_.size() << "\n";

        if (!compiledTypes_.empty()) {
            oss << "Available types: ";
            auto types = getAvailableDataTypes();
            for (size_t i = 0; i < types.size(); ++i) {
              if (i > 0) oss << ", ";
              oss << getDataTypeName(types[i]);
            }
            oss << "\n";
        } else {
            oss << "CRITICAL: No types compiled!\n";
            oss << "This indicates a build configuration error.\n";

            // Show active defines for debugging
            oss << "Active defines:";
#if defined(HAS_BOOL)
            oss << " HAS_BOOL";
#endif
#if defined(HAS_FLOAT)
            oss << " HAS_FLOAT";
#endif
#if defined(HAS_FLOAT32)
            oss << " HAS_FLOAT32";
#endif
#if defined(HAS_DOUBLE)
            oss << " HAS_DOUBLE";
#endif
#if defined(HAS_INT32_T)
            oss << " HAS_INT32_T";
#endif
#if defined(HAS_INT32)
            oss << " HAS_INT32";
#endif
#if defined(HAS_INT)
            oss << " HAS_INT";
#endif
#if defined(HAS_INT64_T)
            oss << " HAS_INT64_T";
#endif
#if defined(HAS_LONG)
            oss << " HAS_LONG";
#endif

            if (compiledTypes_.empty()) {
              oss << "\nNo defines matched - check CMake setup_type_definitions()";
            }
        }

        oss << "==========================================\n";

        return oss.str();
    }
    
    // DataTypeValidator implementation
    void DataTypeValidator::validateOrThrow() {
        if (!DataTypeValidation::isValidDataType(dataType_)) {
            std::string errorMsg = DataTypeValidation::getDataTypeErrorMessage(
                static_cast<int>(dataType_), context_.c_str());
            THROW_EXCEPTION(errorMsg.c_str());
        }
        
        if (!DataTypeValidation::isCompiledDataType(dataType_)) {
            std::string errorMsg = DataTypeValidation::getDataTypeErrorMessage(
                dataType_, context_.c_str());
            THROW_EXCEPTION(errorMsg.c_str());
        }
    }

} // namespace sd
