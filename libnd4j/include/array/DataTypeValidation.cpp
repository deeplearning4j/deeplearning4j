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
        // Auto-register types based on compile-time flags
        // This mirrors the logic from types.h
        
#if defined(HAS_BOOL)
        registerCompiledType(DataType::BOOL, "BOOL");
#endif

#if defined(HAS_FLOAT32)
        registerCompiledType(DataType::FLOAT32, "FLOAT32");
#endif

#if defined(HAS_DOUBLE)
        registerCompiledType(DataType::DOUBLE, "DOUBLE");
#endif

#if defined(HAS_FLOAT16)
        registerCompiledType(DataType::HALF, "HALF");
#endif

#if defined(HAS_BFLOAT16)
        registerCompiledType(DataType::BFLOAT16, "BFLOAT16");
#endif

#if defined(HAS_INT8)
        registerCompiledType(DataType::INT8, "INT8");
#endif

#if defined(HAS_INT16)
        registerCompiledType(DataType::INT16, "INT16");
#endif

#if defined(HAS_INT32)
        registerCompiledType(DataType::INT32, "INT32");
#endif

#if defined(HAS_INT64) || defined(HAS_LONG)
        registerCompiledType(DataType::INT64, "INT64");
#endif

#if defined(HAS_UINT8)
        registerCompiledType(DataType::UINT8, "UINT8");
#endif

#if defined(HAS_UINT16)
        registerCompiledType(DataType::UINT16, "UINT16");
#endif

#if defined(HAS_UINT32)
        registerCompiledType(DataType::UINT32, "UINT32");
#endif

#if defined(HAS_UNSIGNEDLONG)
        registerCompiledType(DataType::UINT64, "UINT64");
#endif

#if defined(HAS_UTF8)
        registerCompiledType(DataType::UTF8, "UTF8");
#endif

#if defined(HAS_UTF16)
        registerCompiledType(DataType::UTF16, "UTF16");
#endif

#if defined(HAS_UTF32)
        registerCompiledType(DataType::UTF32, "UTF32");
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
    
    bool DataTypeValidation::isCompiledDataType(DataType dataType) {
        initializeIfNeeded();
        return compiledTypes_.find(dataType) != compiledTypes_.end();
    }
    
    std::string DataTypeValidation::getDataTypeErrorMessage(DataType dataType, const char* context) {
        return getDataTypeErrorMessage(static_cast<int>(dataType), context);
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
        
        // Add build configuration info
#if defined(SD_SELECTIVE_TYPES)
        oss << "\nBuild configuration: Selective types enabled";
        
        // List the define flags that would enable missing types
        if (isValidDataType(dataTypeId)) {
            DataType dt = enumMapping_[dataTypeId];
            if (!isCompiledDataType(dt)) {
                const char* typeName = getDataTypeName(dt);
                oss << "\nTo enable this type, rebuild with: -DHAS_" << typeName;
            }
        }
#else
        oss << "\nBuild configuration: All types enabled (SD_SELECTIVE_TYPES not defined)";
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
            oss << "\nWARNING: No data types are available in this build!";
        }
        
        // Add suggestions
        if (!isValidDataType(dataTypeId)) {
            oss << "\nSuggestion: Check NDArray creation and data type initialization";
        } else if (!availableTypes.empty()) {
            oss << "\nSuggestion: Use one of the available types or rebuild with the required type enabled";
        }
        
        return oss.str();
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
    
    const char* DataTypeValidation::getDataTypeName(DataType dataType) {
        initializeIfNeeded();
        auto it = typeNames_.find(dataType);
        return (it != typeNames_.end()) ? it->second.c_str() : "UNKNOWN";
    }
    
    std::string DataTypeValidation::getBuildInfo() {
        initializeIfNeeded();
        std::ostringstream oss;
        
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
        } else {
            oss << "WARNING: No types compiled!";
        }
        
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
