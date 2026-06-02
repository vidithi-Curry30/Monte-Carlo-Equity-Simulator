#pragma once
#include <cstring>
#include <type_traits>

// Small-buffer-optimised callable — replaces std::function<void()> in the
// thread pool's task queue.
//
// std::function heap-allocates its stored callable unless the implementation
// applies its own SBO (not guaranteed by the standard, and the buffer size
// varies by toolchain). For a task queue that submits millions of items per
// second, those allocations are measurable.
//
// This class stores callables inline if they fit within kSize bytes, with
// no fallback to the heap. The limit is intentionally strict: if a lambda
// is too large it means the capture is doing too much, and the right fix is
// to pass a pointer to a separately allocated struct, not to silently pay
// for a heap allocation.
//
// One subtlety that's easy to get wrong: move construction can't just
// memcpy the buffer. If the stored type has a non-trivial move constructor
// (e.g. it contains a std::string or another SBO type), memcpy produces a
// second live object pointing at the same internal state. We store a
// separate move_ function pointer to invoke the real move constructor.
// This adds 8 bytes to Task but makes it correct for any movable callable.

class Task {
    static constexpr size_t kSize  = 64;
    static constexpr size_t kAlign = alignof(std::max_align_t);

    using VoidFn = void(*)(void*);
    using MoveFn = void(*)(void*, void*);  // (dst, src)

    alignas(kAlign) char buf_[kSize];
    VoidFn invoke_  = nullptr;
    VoidFn destroy_ = nullptr;
    MoveFn move_    = nullptr;  // see note above

public:
    Task() = default;

    template<typename F,
             typename = std::enable_if_t<!std::is_same_v<std::decay_t<F>, Task>>>
    explicit Task(F&& f) {
        using Fn = std::decay_t<F>;

        static_assert(sizeof(Fn) <= kSize,
            "Callable too large for Task's inline buffer (64 bytes). "
            "Reduce capture size, or wrap the captures in a shared_ptr<State>.");
        static_assert(alignof(Fn) <= kAlign,
            "Callable alignment requirement exceeds Task's buffer alignment.");

        new (buf_) Fn(std::forward<F>(f));
        invoke_  = [](void* p) { (*static_cast<Fn*>(p))(); };
        destroy_ = [](void* p) { static_cast<Fn*>(p)->~Fn(); };
        move_    = [](void* dst, void* src) {
            new (dst) Fn(std::move(*static_cast<Fn*>(src)));
        };
    }

    Task(Task&& o) noexcept
        : invoke_(o.invoke_), destroy_(o.destroy_), move_(o.move_) {
        if (move_) {
            move_(buf_, o.buf_);  // move-construct Fn into our buffer
            o.destroy_(o.buf_);   // destroy the moved-from object in o's buffer
            o.invoke_ = o.destroy_ = nullptr; o.move_ = nullptr;
        }
    }

    Task& operator=(Task&& o) noexcept {
        if (this != &o) {
            reset();
            invoke_ = o.invoke_; destroy_ = o.destroy_; move_ = o.move_;
            if (move_) {
                move_(buf_, o.buf_);
                o.destroy_(o.buf_);  // same: destroy moved-from object
                o.invoke_ = o.destroy_ = nullptr; o.move_ = nullptr;
            }
        }
        return *this;
    }

    Task(const Task&)            = delete;
    Task& operator=(const Task&) = delete;

    ~Task() { reset(); }

    void operator()() { invoke_(buf_); }
    explicit operator bool() const { return invoke_ != nullptr; }

private:
    void reset() {
        if (destroy_) {
            destroy_(buf_);
            invoke_ = destroy_ = nullptr;
            move_ = nullptr;
        }
    }
};
