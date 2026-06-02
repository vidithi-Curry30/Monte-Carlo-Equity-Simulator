#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include "task.hpp"

// Hand-rolled fixed-size thread pool.
//
// Design decisions worth knowing:
//
// 1. Tasks are stored as Task (src/task.hpp), not std::function. Task uses
//    a 64-byte inline buffer so submission never allocates. std::function's
//    SBO is implementation-defined and its heap fallback showed up in
//    profiles when submitting millions of short-lived tasks.
//
// 2. pending_ is std::atomic<int>, not a plain int under the task mutex.
//    Incrementing before enqueue and decrementing after task completion
//    means wait() can safely check without holding the task lock.
//
// 3. Two condition variables: cv_ wakes workers when work arrives or the
//    pool is stopping; done_cv_ wakes the caller of wait() when pending_
//    hits zero. A single cv_ would require the caller to re-acquire the
//    task mutex on every completion, adding unnecessary contention.
//
// 4. stop_ is read under mutex_ so workers see it correctly when they
//    wake from cv_.wait(). No separate atomic needed.
//
// 5. RAII: the destructor sets stop_, drains sleeping workers via
//    notify_all(), then joins every thread. No detached threads, no leaks.
//
// Known limitation: the internal queue is unbounded. For this project
// (submit N tasks, wait, repeat) that's fine — the caller naturally
// throttles submission. A general-purpose pool would need a capacity
// limit and backpressure.

class ThreadPool {
public:
    explicit ThreadPool(size_t n_threads) : pending_(0), stop_(false) {
        workers_.reserve(n_threads);
        for (size_t i = 0; i < n_threads; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    // Non-copyable, non-movable — threads hold a pointer to this object.
    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template<typename F>
    void submit(F&& f) {
        // Increment before enqueue: ensures pending_ > 0 the moment the
        // task exists, so wait() cannot spuriously return zero.
        pending_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace(std::forward<F>(f));
        }
        cv_.notify_one();
    }

    // Block until all submitted tasks have completed.
    void wait() {
        std::unique_lock<std::mutex> lock(done_mutex_);
        done_cv_.wait(lock, [this] {
            return pending_.load(std::memory_order_acquire) == 0;
        });
    }

    size_t thread_count() const { return workers_.size(); }

private:
    void worker_loop() {
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
            // fetch_sub returns the value before subtraction.
            // If it was 1, we just hit zero — wake any waiting caller.
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                done_cv_.notify_all();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<Task>         tasks_;
    std::mutex                      mutex_;
    std::condition_variable         cv_;
    std::mutex                      done_mutex_;
    std::condition_variable         done_cv_;
    std::atomic<int>                pending_;
    bool                            stop_;
};
