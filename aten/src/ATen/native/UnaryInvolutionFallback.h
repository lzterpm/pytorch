#pragma once

#include <ATen/native/TransformFallback.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <torch/library.h>

namespace at { class Tensor; }
namespace c10 { class OperatorHandle; }

namespace at::native {

// Register a dispatch for a "simple" dispatch based transformation
// with the following requirements:
//  * is an involution (i.e. it's own inverse)
//  * is unary
//  * requires no state other than whether or not the transformation
//    is applied to the tensor
//  * the transformation is implemented within copy_
template <c10::DispatchKey key>
auto register_unary_involution_fallback(torch::Library& library);

namespace detail {

class UnaryInvolutionFallback final : public TransformFallback {
 public:
  explicit UnaryInvolutionFallback(c10::DispatchKey key) : TransformFallback(key) {}
  ~UnaryInvolutionFallback() final;

 private:
  auto transform(Tensor const& tensor) const -> Tensor final;
  auto untransform(Tensor& output, Tensor const& result) const -> void final;
};

template <c10::DispatchKey key>
void unary_involution_fallback_trampoline(c10::OperatorHandle const& op, c10::DispatchKeySet dispatch_keys,
                                          torch::jit::Stack* stack) {
  UnaryInvolutionFallback{key}(op, dispatch_keys, stack);
}

} // namespace detail

template <c10::DispatchKey key>
auto register_unary_involution_fallback(torch::Library& library) {
  library.fallback(torch::CppFunction::makeFromBoxedFunction<&detail::unary_involution_fallback_trampoline<key>>());
}

} // namespace at::native
