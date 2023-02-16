# pragma once

#include <string_view>

#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/Stack.h>

// Represents a dispatch key that transforms a tensor on entry into
// and exit from an operator. Examples include structure preserving
// transforms (e.g. functors) such as negation and conjugation, as
// well as structure modifying transforms like compound view.

namespace at { class Tensor; }
namespace c10 { class OperatorHandle; }
namespace torch { class Library; }

namespace at::native {

class TransformFallback {
 public:
  explicit TransformFallback(c10::DispatchKey key) : key_(key) {}
  virtual ~TransformFallback();

  auto operator()(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, c10::Stack* stack)
    -> void;

 protected:
  // Applies the transform to the input.
  virtual auto transform(Tensor const& tensor) const -> Tensor = 0;

  // Untransforms result from the operator into output.
  virtual auto untransform(Tensor& output, Tensor const& result) const -> void = 0;

 private:
  auto has_key(const Tensor& tensor) const -> bool;

  auto operator_name() const -> std::string_view;

  c10::DispatchKey key_;
};

} // namespace at::native
