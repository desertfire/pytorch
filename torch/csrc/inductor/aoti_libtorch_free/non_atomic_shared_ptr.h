#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace aoti::libtorch_free {

template <typename T>
class NonAtomicSharedPtr {
 private:
  struct ControlBlock {
    int count = 1;
    T* ptr;
    using Deleter = void (*)(T*);
    Deleter deleter;

    ControlBlock(T* p, Deleter d) : ptr(p), deleter(d) {}
    ControlBlock(const ControlBlock&) = delete;
    ControlBlock& operator=(const ControlBlock&) = delete;
    ControlBlock(ControlBlock&&) = delete;
    ControlBlock& operator=(ControlBlock&&) = delete;

    ~ControlBlock() {
      if (ptr) {
        deleter(ptr);
      }
    }
  };

  ControlBlock* cb_;

  void cleanup() {
    if (cb_ && --cb_->count == 0) {
      delete cb_;
    }
    cb_ = nullptr;
  }

 public:
  // Default constructor
  NonAtomicSharedPtr() noexcept : cb_(nullptr) {}

  // Constructor from raw pointer
  explicit NonAtomicSharedPtr(
      T* p,
      typename ControlBlock::Deleter d = [](T* p) { delete p; })
      : cb_(p ? new ControlBlock(p, d) : nullptr) {}

  // Copy constructor
  NonAtomicSharedPtr(const NonAtomicSharedPtr& other) noexcept
      : cb_(other.cb_) {
    if (cb_) {
      ++cb_->count;
    }
  }

  // Move constructor
  NonAtomicSharedPtr(NonAtomicSharedPtr&& other) noexcept : cb_(other.cb_) {
    other.cb_ = nullptr;
  }

  // Destructor
  ~NonAtomicSharedPtr() {
    cleanup();
  }

  // Copy assignment
  NonAtomicSharedPtr& operator=(const NonAtomicSharedPtr& other) noexcept {
    if (this != &other) {
      cleanup();
      cb_ = other.cb_;
      if (cb_) {
        ++cb_->count;
      }
    }
    return *this;
  }

  // Move assignment
  NonAtomicSharedPtr& operator=(NonAtomicSharedPtr&& other) noexcept {
    if (this != &other) {
      cleanup();
      cb_ = other.cb_;
      other.cb_ = nullptr;
    }
    return *this;
  }

  // Modifiers
  void reset(T* p = nullptr, typename ControlBlock::Deleter d = {}) {
    *this = NonAtomicSharedPtr(p, d);
  }

  void swap(NonAtomicSharedPtr& other) noexcept {
    std::swap(cb_, other.cb_);
  }

  // Observers
  T* get() const noexcept {
    return cb_ ? cb_->ptr : nullptr;
  }
  T& operator*() const {
    if (!cb_) {
      throw std::runtime_error("Dereferencing null NonAtomicSharedPtr");
    }
    return *cb_->ptr;
  }
  T* operator->() const {
    if (!cb_) {
      throw std::runtime_error("Accessing member of null NonAtomicSharedPtr");
    }
    return cb_->ptr;
  }
  long use_count() const noexcept {
    return cb_ ? cb_->count : 0;
  }
  explicit operator bool() const noexcept {
    return cb_ != nullptr;
  }

  // Friend swap for ADL
  friend void swap(NonAtomicSharedPtr& a, NonAtomicSharedPtr& b) noexcept {
    a.swap(b);
  }
};

} // namespace aoti::libtorch_free
