#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Lock-free single-producer / single-consumer queue.
//
// Intended use: a dedicated pipeline with one thread writing and one thread
// reading — e.g. a market data feed thread pushing ticks, and a strategy
// thread consuming them. Not suitable as the thread pool's task queue
// (multiple workers pop from that), but for SPSC pipelines it eliminates
// the mutex entirely.
//
// Implementation notes:
//
// Indices are uint64_t and never wrap. At one billion enqueues per second —
// faster than anything this process will do — overflow takes ~585 years.
// This avoids the modular arithmetic edge cases that bite uint32_t queues.
//
// head_ and tail_ are on separate cache lines (alignas(64)). The producer
// only writes tail_ and reads head_; the consumer only writes head_ and
// reads tail_. If they shared a line, every enqueue/dequeue would bounce
// the line between cores even with no contention on the data itself.
//
// Capacity must be a power of 2. This lets us replace modulo with a
// bitmask (idx & MASK), which the CPU can fold into an addressing mode.
// We assert at construction rather than silently rounding up — rounding
// would make the actual capacity surprising.
//
// Known limitation: push() spins if the queue is full. For our use case
// (bursty but bounded) this doesn't happen in practice. If backpressure
// matters, the caller should check try_push()'s return value and apply
// flow control upstream.
//
// TODO: on multi-socket NUMA machines, producer and consumer on different
// nodes will see ~2x higher latency accessing the slots_ array due to
// remote memory. A per-NUMA-node queue pair would fix this but adds
// significant complexity; skip it until profiling shows it's the bottleneck.

template<typename T, size_t Capacity>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0,
        "SPSCQueue capacity must be a power of 2.");
    static_assert(std::is_default_constructible_v<T>,
        "T must be default-constructible (slots are pre-allocated).");

    static constexpr size_t kMask = Capacity - 1;

public:
    SPSCQueue()  = default;
    ~SPSCQueue() = default;

    // Non-copyable, non-movable. The atomics have no meaningful copy
    // semantics, and moving a live queue would leave a dangling reader.
    SPSCQueue(const SPSCQueue&)            = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    // Returns false if the queue is full. Does not block.
    bool try_push(const T& val) {
        const uint64_t t = tail_.load(std::memory_order_relaxed);
        if (t - head_.load(std::memory_order_acquire) == Capacity)
            return false;
        slots_[t & kMask] = val;
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }

    bool try_push(T&& val) {
        const uint64_t t = tail_.load(std::memory_order_relaxed);
        if (t - head_.load(std::memory_order_acquire) == Capacity)
            return false;
        slots_[t & kMask] = std::move(val);
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }

    // Returns false if the queue is empty. Does not block.
    // Out-param instead of optional<T>: avoids requiring T to be
    // move-constructible into an optional, and makes the no-item
    // path zero-cost (no optional construction).
    bool try_pop(T& out) {
        const uint64_t h = head_.load(std::memory_order_relaxed);
        if (tail_.load(std::memory_order_acquire) == h)
            return false;
        out = std::move(slots_[h & kMask]);
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    // Approximate. In a concurrent setting, the true size can change
    // between the two atomic loads. Only use this for diagnostics, never
    // for correctness decisions.
    size_t size_approx() const {
        const uint64_t h = head_.load(std::memory_order_relaxed);
        const uint64_t t = tail_.load(std::memory_order_relaxed);
        return (t > h) ? static_cast<size_t>(t - h) : 0;
    }

    static constexpr size_t capacity() { return Capacity; }

private:
    // Slots are pre-allocated in the object itself — no heap, no indirection.
    T slots_[Capacity];

    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) std::atomic<uint64_t> tail_{0};
};
