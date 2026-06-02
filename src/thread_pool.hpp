#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// Hand-rolled fixed-size thread pool.
//
// Design decisions worth knowing:
//
// 1. pending_ is std::atomic<int>, not a plain int under the task mutex.
//    Incrementing before enqueue and decrementing after task completion
//    means wait() can safely spin-check without holding the task lock.
//
// 2. Two condition variables: cv_ wakes workers when work arrives or the
//    pool is stopping; done_cv_ wakes the caller of wait() when pending_
//    hits zero. A single cv_ would require the caller to re-acquire the
//    task mutex on every task completion, creating unnecessary contention.
//
// 3. stop_ is read under mutex_ so workers see it correctly when they
//    wake from cv_.wait(). No separate atomic needed.
//
// 4. RAII: the destructor sets stop_, drains sleeping workers via
//    notify_all(), then joins every thread. No detached threads, no leaks.

class ThreadPool {
public:
    explicit ThreadPool(size_t n_threads) : stop_(false), pending_(0) {
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
            std::function<void()> task;
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

    std::vector<std::thread>        workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex                      mutex_;
    std::condition_variable         cv_;
    std::mutex                      done_mutex_;
    std::condition_variable         done_cv_;
    std::atomic<int>                pending_;
    bool                            stop_;
};
