// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// TODO(wesm): LLVM 7 produces pesky C4244 that disable pragmas around the LLVM
// includes seem to not fix as with LLVM 6
#if defined(_MSC_VER)
#pragma warning(disable : 4244)
#endif

#include "gandiva/engine.h"

#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include "arrow/util/logging.h"
#include "emscripten.h"
#include "gandiva/selection_vector.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4141)
#pragma warning(disable : 4146)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#pragma warning(disable : 4624)
#endif

#include <llvm/Analysis/Passes.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/SubtargetFeature.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Vectorize.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "arrow/util/make_unique.h"
#include "gandiva/configuration.h"
#include "gandiva/decimal_ir.h"
#include "gandiva/exported_funcs_registry.h"

namespace gandiva {

extern const unsigned char kPrecompiledBitcode[];
extern const size_t kPrecompiledBitcodeSize;

std::once_flag llvm_init_once_flag;
static bool llvm_init = false;
static llvm::StringRef cpu_name;
static llvm::SmallVector<std::string, 10> cpu_attrs;

std::unique_ptr<llvm::TargetMachine> createTargetMachine() {
  auto TT(llvm::Triple::normalize("wasm32-unknown-unknown"));
  std::string CPU("");
  std::string FS("");

  LLVMInitializeWebAssemblyTargetInfo();
  LLVMInitializeWebAssemblyTarget();
  LLVMInitializeWebAssemblyTargetMC();

  std::string Error;
  const llvm::Target* TheTarget = llvm::TargetRegistry::lookupTarget(TT, Error);
  assert(TheTarget);

  return std::unique_ptr<llvm::TargetMachine>(static_cast<llvm::TargetMachine*>(
      TheTarget->createTargetMachine(TT, CPU, FS, llvm::TargetOptions(), llvm::None,
                                     llvm::None, llvm::CodeGenOpt::Default)));
}

void Engine::InitOnce() {
  std::cout << "Engine::InitOnce 1" << std::endl;
  DCHECK_EQ(llvm_init, false);
  std::cout << "Engine::InitOnce 2" << std::endl;

  LLVMInitializeWebAssemblyTargetInfo();
  LLVMInitializeWebAssemblyTarget();
  LLVMInitializeWebAssemblyTargetMC();
  std::cout << "Engine::InitOnce 3" << std::endl;
  LLVMInitializeWebAssemblyAsmPrinter();
  std::cout << "Engine::InitOnce 4" << std::endl;
  LLVMInitializeWebAssemblyAsmParser();
  std::cout << "Engine::InitOnce 5" << std::endl;
  LLVMInitializeWebAssemblyDisassembler();
  std::cout << "Engine::InitOnce 6" << std::endl;
  // llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  std::cout << "Engine::InitOnce 7" << std::endl;

  cpu_name = llvm::sys::getHostCPUName();
  llvm::StringMap<bool> host_features;
  std::string cpu_attrs_str;
  if (llvm::sys::getHostCPUFeatures(host_features)) {
    for (auto& f : host_features) {
      std::string attr = f.second ? std::string("+") + f.first().str()
                                  : std::string("-") + f.first().str();
      cpu_attrs.push_back(attr);
      cpu_attrs_str += " " + attr;
    }
  }
  ARROW_LOG(INFO) << "Detected CPU Name : " << cpu_name.str();
  ARROW_LOG(INFO) << "Detected CPU Features:" << cpu_attrs_str;
  llvm_init = true;
}

Engine::Engine(const std::shared_ptr<Configuration>& conf,
               std::unique_ptr<llvm::LLVMContext> ctx,
               std::unique_ptr<llvm::ExecutionEngine> engine, llvm::Module* module)
    : context_(std::move(ctx)),
      execution_engine_(std::move(engine)),
      ir_builder_(arrow::internal::make_unique<llvm::IRBuilder<>>(*context_)),
      module_(module),
      types_(*context_),
      optimize_(conf->optimize()) {}

Status Engine::Init() {
  // Add mappings for functions that can be accessed from LLVM/IR module.
  AddGlobalMappings();

  ARROW_RETURN_NOT_OK(LoadPreCompiledIR());
  ARROW_RETURN_NOT_OK(DecimalIR::AddFunctions(this));

  return Status::OK();
}

/// factory method to construct the engine.
Status Engine::Make(const std::shared_ptr<Configuration>& conf,
                    std::unique_ptr<Engine>* out) {
  std::cout << "Engine::Make 1" << std::endl;
  std::call_once(llvm_init_once_flag, InitOnce);
  std::cout << "Engine::Make 2" << std::endl;

  auto ctx = arrow::internal::make_unique<llvm::LLVMContext>();
  std::cout << "Engine::Make 3" << std::endl;
  auto module = arrow::internal::make_unique<llvm::Module>("codegen", *ctx);
  std::cout << "Engine::Make 4" << std::endl;

  // Capture before moving, ExecutionEngine does not allow retrieving the
  // original Module.
  auto module_ptr = module.get();
  std::cout << "Engine::Make 5" << std::endl;

  auto opt_level =
      conf->optimize() ? llvm::CodeGenOpt::Aggressive : llvm::CodeGenOpt::None;

  // Note that the lifetime of the error string is not captured by the
  // ExecutionEngine but only for the lifetime of the builder. Found by
  // inspecting LLVM sources.
  std::string builder_error;

  std::cout << "Engine::Make 6" << std::endl;

  llvm::EngineBuilder engine_builder(std::move(module));

  std::cout << "Engine::Make 7" << std::endl;

  engine_builder.setEngineKind(llvm::EngineKind::JIT)
      .setOptLevel(opt_level)
      .setErrorStr(&builder_error);

  std::cout << "Engine::Make 8, cpu_name: " << cpu_name.str() << ", cpu_attrs: ";

  for (auto& attr : cpu_attrs) {
    std::cout << attr << ", ";
  }

  std::cout << std::endl;

  if (conf->target_host_cpu()) {
    std::cout << "Engine::Make 9" << std::endl;
    engine_builder.setMCPU(cpu_name);
    engine_builder.setMAttrs(cpu_attrs);
  }

  std::cout << "Engine::Make 10" << std::endl;
  auto tm = createTargetMachine();

  std::cout << "Engine::Make 10.0 tm" << tm << std::endl;
  auto& triple = tm->getTargetTriple();
  std::cout << "Engine::Make 10.0 triple.getArch()" << triple.getArch() << std::endl;
  std::cout << "Engine::Make 10.0 triple.getSubArch()" << triple.getSubArch()
            << std::endl;
  std::cout << "Engine::Make 10.0 triple.getVendor()" << triple.getVendor() << std::endl;
  std::cout << "Engine::Make 10.0 triple.getOS()" << triple.getOS() << std::endl;

  std::unique_ptr<llvm::ExecutionEngine> exec_engine{engine_builder.create(tm.release())};

  std::cout << "Engine::Make 11" << std::endl;

  if (exec_engine == nullptr) {
    return Status::CodeGenError("Could not instantiate llvm::ExecutionEngine: ",
                                builder_error);
  }

  std::cout << "Engine::Make 12" << std::endl;

  std::unique_ptr<Engine> engine{
      new Engine(conf, std::move(ctx), std::move(exec_engine), module_ptr)};

  std::cout << "Engine::Make 13" << std::endl;
  ARROW_RETURN_NOT_OK(engine->Init());

  std::cout << "Engine::Make 14" << std::endl;
  *out = std::move(engine);
  return Status::OK();
}

// This method was modified from its original version for a part of MLIR
// Original source from
// https://github.com/llvm/llvm-project/blob/9f2ce5b915a505a5488a5cf91bb0a8efa9ddfff7/mlir/lib/ExecutionEngine/ExecutionEngine.cpp
// The original copyright notice follows.

// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

static void SetDataLayout(llvm::Module* module) {
  auto target_triple = std::string("wasm32-unknown-unknown-wasm");
  // auto target_triple = llvm::sys::getDefaultTargetTriple();
  std::string error_message;
  auto target = llvm::TargetRegistry::lookupTarget(target_triple, error_message);
  if (!target) {
    return;
  }

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> host_features;

  if (llvm::sys::getHostCPUFeatures(host_features)) {
    for (auto& f : host_features) {
      features.AddFeature(f.first(), f.second);
    }
  }

  std::unique_ptr<llvm::TargetMachine> machine(
      target->createTargetMachine(target_triple, cpu, features.getString(), {}, {}));

  module->setDataLayout(machine->createDataLayout());
}
// end of the mofified method from MLIR

// Handling for pre-compiled IR libraries.
Status Engine::LoadPreCompiledIR() {
  std::cout << "Engine::LoadPreCompiledIR 1" << std::endl;
  auto bitcode = llvm::StringRef(reinterpret_cast<const char*>(kPrecompiledBitcode),
                                 kPrecompiledBitcodeSize);
  std::cout << "Engine::LoadPreCompiledIR 2" << std::endl;
  /// Read from file into memory buffer.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer_or_error =
      llvm::MemoryBuffer::getMemBuffer(bitcode, "precompiled", false);

  std::cout << "Engine::LoadPreCompiledIR 3" << std::endl;

  ARROW_RETURN_IF(!buffer_or_error,
                  Status::CodeGenError("Could not load module from IR: ",
                                       buffer_or_error.getError().message()));

  std::cout << "Engine::LoadPreCompiledIR 4" << std::endl;

  std::unique_ptr<llvm::MemoryBuffer> buffer = move(buffer_or_error.get());

  std::cout << "Engine::LoadPreCompiledIR 5" << std::endl;

  /// Parse the IR module.
  llvm::Expected<std::unique_ptr<llvm::Module>> module_or_error =
      llvm::getOwningLazyBitcodeModule(move(buffer), *context());

  std::cout << "Engine::LoadPreCompiledIR 6" << std::endl;

  if (!module_or_error) {
    std::cout << "Engine::LoadPreCompiledIR 7" << std::endl;
    // NOTE: llvm::handleAllErrors() fails linking with RTTI-disabled LLVM builds
    // (ARROW-5148)
    std::string str;
    llvm::raw_string_ostream stream(str);
    stream << module_or_error.takeError();
    return Status::CodeGenError(stream.str());
  }

  std::cout << "Engine::LoadPreCompiledIR 8" << std::endl;
  std::unique_ptr<llvm::Module> ir_module = move(module_or_error.get());

  std::cout << "Engine::LoadPreCompiledIR 9" << std::endl;

  // set dataLayout
  SetDataLayout(ir_module.get());

  std::cout << "Engine::LoadPreCompiledIR 10" << std::endl;

  ARROW_RETURN_IF(llvm::verifyModule(*ir_module, &llvm::errs()),
                  Status::CodeGenError("verify of IR Module failed"));

  std::cout << "Engine::LoadPreCompiledIR 11" << std::endl;

  ARROW_RETURN_IF(llvm::Linker::linkModules(*module_, move(ir_module)),
                  Status::CodeGenError("failed to link IR Modules"));

  std::cout << "Engine::LoadPreCompiledIR 12" << std::endl;

  return Status::OK();
}

// Get rid of all functions that don't need to be compiled.
// This helps in reducing the overall compilation time. This pass is trivial,
// and is always done since the number of functions in gandiva is very high.
// (Adapted from Apache Impala)
//
// Done by marking all the unused functions as internal, and then, running
// a pass for dead code elimination.
Status Engine::RemoveUnusedFunctions() {
  // Setup an optimiser pipeline
  std::unique_ptr<llvm::legacy::PassManager> pass_manager(
      new llvm::legacy::PassManager());

  std::unordered_set<std::string> used_functions;
  used_functions.insert(functions_to_compile_.begin(), functions_to_compile_.end());

  pass_manager->add(
      llvm::createInternalizePass([&used_functions](const llvm::GlobalValue& func) {
        return (used_functions.find(func.getName().str()) != used_functions.end());
      }));
  pass_manager->add(llvm::createGlobalDCEPass());
  pass_manager->run(*module_);
  return Status::OK();
}

// Optimise and compile the module.
Status Engine::FinalizeModule() {
  std::cout << "Engine::FinalizeModule 1" << std::endl;
  ARROW_RETURN_NOT_OK(RemoveUnusedFunctions());
  std::cout << "Engine::FinalizeModule 2" << std::endl;

  if (optimize_) {
    std::cout << "Engine::FinalizeModule optimize 1" << std::endl;
    // misc passes to allow for inlining, vectorization, ..
    std::unique_ptr<llvm::legacy::PassManager> pass_manager(
        new llvm::legacy::PassManager());

    llvm::TargetIRAnalysis target_analysis =
        execution_engine_->getTargetMachine()->getTargetIRAnalysis();
    pass_manager->add(llvm::createTargetTransformInfoWrapperPass(target_analysis));
    pass_manager->add(llvm::createFunctionInliningPass());
    pass_manager->add(llvm::createInstructionCombiningPass());
    pass_manager->add(llvm::createPromoteMemoryToRegisterPass());
    pass_manager->add(llvm::createGVNPass());
    pass_manager->add(llvm::createNewGVNPass());
    pass_manager->add(llvm::createCFGSimplificationPass());
    pass_manager->add(llvm::createLoopVectorizePass());
    pass_manager->add(llvm::createSLPVectorizerPass());
    pass_manager->add(llvm::createGlobalOptimizerPass());

    // run the optimiser
    llvm::PassManagerBuilder pass_builder;
    pass_builder.OptLevel = 3;
    pass_builder.populateModulePassManager(*pass_manager);
    pass_manager->run(*module_);
  }

  std::cout << "Engine::FinalizeModule 3" << std::endl;
  ARROW_RETURN_IF(llvm::verifyModule(*module_, &llvm::errs()),
                  Status::CodeGenError("Module verification failed after optimizer"));
  std::cout << "Engine::FinalizeModule 4" << std::endl;

  module_->print(llvm::errs(), nullptr);

  // do the compilation
  execution_engine_->finalizeObject();
  std::cout << "Engine::FinalizeModule 5" << std::endl;
  module_finalized_ = true;

  return Status::OK();
}

void* Engine::CompiledFunction(llvm::Function* irFunction) {
  DCHECK(module_finalized_);
  std::cout << "Engine::CompiledFunction" << std::endl;
  return execution_engine_->getPointerToFunction(irFunction);
}

void Engine::SetCompiledFunction(llvm::Function* irFunction, SelectionVector::Mode mode) {
  std::cout << "Engine::SetCompiledFunction" << std::endl;
  execution_engine_->generateCodeForModule(irFunction->getParent());
  EM_ASM(
      {
        console.log("Setting compiled function");
        const bitcode = FS.readFile("/jit.wasm");
        const module = new WebAssembly.Module(bitcode);
        const instance = new WebAssembly.Instance(
            module, {env : {__linear_memory : window.wasmMemory}});
        console.log("Setting function on index", $0, instance.exports._start);
        window.jitFunctions[$0] = instance.exports._start;
      },
      static_cast<int>(mode));
}

void Engine::AddGlobalMappingForFunc(const std::string& name, llvm::Type* ret_type,
                                     const std::vector<llvm::Type*>& args,
                                     void* function_ptr) {
  constexpr bool is_var_arg = false;
  auto prototype = llvm::FunctionType::get(ret_type, args, is_var_arg);
  constexpr auto linkage = llvm::GlobalValue::ExternalLinkage;
  auto fn = llvm::Function::Create(prototype, linkage, name, module());
  execution_engine_->addGlobalMapping(fn, function_ptr);
}

void Engine::AddGlobalMappings() { ExportedFuncsRegistry::AddMappings(this); }

std::string Engine::DumpIR() {
  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module_->print(stream, nullptr);
  return ir;
}

}  // namespace gandiva
