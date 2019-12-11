/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2017 by Contributors
 * \file vhls_module.cc
 */
#include "vhls_module.h"

#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include <unordered_map>
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "../meta_data.h"
#include "../file_util.h"
#include "../module_util.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tvm {
namespace runtime {

// a wrapped function class to get packed func.
class VHLSWrappedFunc {
public:
  // initialize the CUDA function.
  VHLSWrappedFunc(void (*func_addr)(void*)) : func_addr_(func_addr) 
  {}
  // invoke the function with void arguments
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void* void_args,
                  size_t packed_nbytes) const {
    func_addr_(void_args);
  }

 private:
  // The name of the function.
  void (*func_addr_)(void*);
};

// Module to support thread-safe multi-GPU execution.
// cuModule is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class VHLSModuleNode : public runtime::ModuleNode {
 public:
  explicit VHLSModuleNode(std::string data,
                          std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string source,
                          std::string fname)
    : data_(data), fmt_(fmt), fmap_(fmap), source_(source) {
    if (fmt_ == "so") {
      Load(fname);
    }
  }
  
  void Init(std::string fname) {
    if (fmt_ == "so") {
      Load(fname);
    }
  }

  const char* type_key() const final {
    return "vhls";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    if (fmt_ == "so") {
      // Csimulation
      CHECK_EQ(sptr_to_self.get(), this);
      CHECK_NE(name, symbol::tvm_module_main)
          << "Device function do not have main";
      // const BackendPackedCFunc faddr = reinterpret_cast<BackendPackedCFunc>(GetSymbol(name.c_str()));
      void* faddr = GetSymbol(name.c_str());
      auto it = fmap_.find(name);
      
      if (it == fmap_.end()) {
        return PackedFunc();
      }

      const FunctionInfo& info = it->second;
      VHLSWrappedFunc f((void(*)(void*))faddr);
      return PackFuncPackedArg(f, info.arg_types);
      // if (faddr == nullptr) return PackedFunc();
      // return WrapPackedFunc(faddr, sptr_to_self);
    } else {
      LOG(FATAL) << "not implemented";
    }
  }
  
  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "cc") {
      CHECK_NE(source_.length(), 0);
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, source_);
    } else {
      CHECK_EQ(fmt, fmt_)
          << "Can only save to format=" << fmt_;
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, data_);
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

  std::string GetSource(const std::string& format) final {
    return source_;
  }

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The VHLS source.
  std::string source_;

 private:
  // Platform dependent handling.
#if defined(_WIN32)
  // library handle
  HMODULE lib_handle_{nullptr};
  // Load the library
  void Load(const std::string& name) {
    // use wstring version that is needed by LLVM.
    std::wstring wname(name.begin(), name.end());
    lib_handle_ = LoadLibraryW(wname.c_str());
    CHECK(lib_handle_ != nullptr)
        << "Failed to load dynamic shared library " << name;
  }
  void* GetSymbol(const char* name) {
    return reinterpret_cast<void*>(
        GetProcAddress(lib_handle_, (LPCSTR)name)); // NOLINT(*)
  }
  void Unload() {
    FreeLibrary(lib_handle_);
  }
#else
  // Library handle
  void* lib_handle_{nullptr};
  // load the library
  void Load(const std::string& name) {
    lib_handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
    CHECK(lib_handle_ != nullptr)
        << "Failed to load dynamic shared library " << name
        << " " << dlerror();
  }
  void* GetSymbol(const char* name) {
    return dlsym(lib_handle_, name);
  }
  void Unload() {
    dlclose(lib_handle_);
  }
#endif
};

Module VHLSModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string source,
    std::string fname) {
  std::shared_ptr<VHLSModuleNode> n =
      std::make_shared<VHLSModuleNode>(data, fmt, fmap, source, fname);
  return Module(n);
}

// Load module from module.
Module VHLSModuleLoadFile(const std::string& file_name,
                          const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return VHLSModuleCreate(data, fmt, fmap, std::string(), file_name);
}

Module VHLSModuleLoadBinary(void* strm,
                            const std::string& fname) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return VHLSModuleCreate(data, fmt, fmap, std::string(), "/home/takato.yamada/tkhome/ghq/gitlab.fixstars.com/GENESIS/DNN/tvm/tests/python/contrib/vhls/addone_kernel0.so");
}

TVM_REGISTER_GLOBAL("module.loadfile_vhlsso")
.set_body_typed(VHLSModuleLoadFile);

TVM_REGISTER_GLOBAL("module.loadbinary_vhls")
.set_body_typed(VHLSModuleLoadBinary);
}  // namespace runtime
}  // namespace tvm
