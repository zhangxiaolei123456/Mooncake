// Copyright 2025 Bytedance Technologies Co., Ltd
// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "transport/tcp_transport/tcp_transport_performance.h"
#include "cuda_alike.h"

namespace mooncake {
TcpTransportPerformance::TcpTransportPerformance()
    : transport_(std::make_unique<TcpTransport>())  {
}

TcpTransportPerformance::~TcpTransportPerformance() {
    running_ = false;
    transfer_cond_.notify_one();
    transferThread_.join();
    free(hostAddr_);
    cudaFree(devAddr_);
    hostAddr_ = nullptr;
    devAddr_ = nullptr;
}

void TcpTransportPerformance::transferLoop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(transfer_mutex_);
            transfer_cond_.wait(lock, [this] { return !transferQueues_.empty() || !running_; });
            if (transferQueues_.empty()) {
                continue;
            }

            auto pkg = std::move(transferQueues_.front());
            transferQueues_.pop();
            auto &task_list = pkg.tasks;
            lock.unlock();
            if (task_list.empty()) {
                LOG(ERROR)
                    << "TcpTransportPerformance: empty transfer task batch";
                continue;
            }
            uint64_t total_length = pkg.total_length;
            if ((offset_ + total_length) > HUGE_HOST_SIZE) {
                offset_ = 0;
            }

            cudaError_t cuda_status = cudaMemcpy(static_cast<char *>(hostAddr_) + offset_, hugeDevAddrs[pkg.devId],
                           total_length, cudaMemcpyDefault);
            if (cuda_status != cudaSuccess) {
                LOG(ERROR) << "TcpTransportPerformance: cudaMemcpy dtoh "
                          "error, ret: "
                       << cuda_status << ", hostAddr: " << hostAddr_
                       << ", offset_: " << offset_
                       << ", deviceAddr: " << hugeDevAddrs[pkg.devId]
                       << "len: " << total_length;
                transfer_counter_.fetch_add(1);
                continue;
            }

            for (size_t index = 0; index < task_list.size(); ++index) {
                auto &task = *task_list[index];
                auto &request = *task.request;
                request.source = static_cast<char *>(hostAddr_) + offset_;
                offset_ += request.length;
            }

            std::unique_lock<std::mutex> lock_dev(dev_mtx_);
            mem_blocks[pkg.devId] = false;
            dev_cv_.notify_one();
            Status s = transport_->submitTransferTask(task_list);
            if (!s.ok()) {
                LOG(ERROR)<< "TcpTransportPerformance: Tcp submitTransferTask error";
            }
            transfer_counter_.fetch_add(1);
        }
    }

int TcpTransportPerformance::install(std::string &local_server_name,
                                        std::shared_ptr<TransferMetadata> meta,
                                        std::shared_ptr<Topology> topo) {
    local_server_name_ = local_server_name;
    running_ = true;
    metadata_ = meta;
    hostAddr_ = static_cast<char*>(aligned_alloc(64, HUGE_HOST_SIZE));
    if (hostAddr_ == nullptr) {
        LOG(ERROR) << "TcpTransportPerformance:hostAddr_ is null, "
                      "aligned_alloc failed";
        return -1;
    }
    devAddr_ = nullptr;
    int ret = cudaMalloc(&devAddr_, HUGE_DEVICE_NUM * HUGE_DEVICE_SIZE);

    for (int i = 0; i < HUGE_DEVICE_NUM; i++) {
        hugeDevAddrs.push_back(static_cast<char *>(devAddr_) +
                               i * HUGE_DEVICE_SIZE);
    }

    transferThread_ =
        std::thread(&TcpTransportPerformance::transferLoop, this);

    ret = transport_->install(local_server_name_, meta, topo);
    if (ret) {
        LOG(ERROR) << "TcpTransportPerformance::TcpTransport install error, ret: " << ret;
        return ret;
    }
    ret = transport_->registerLocalMemory(hostAddr_, HUGE_HOST_SIZE, "cpu",
                                          true, true);
    if (ret) {
        LOG(ERROR)
            << "TcpTransportPerformance: registerLocalMemory error, ret: "
            << ret;
        return ret;
    }
    LOG(INFO) << "TcpTransportPerformance install ok";
    return ret;
}

int TcpTransportPerformance::registerLocalMemory(void *addr, size_t length,
                                                    const std::string &name,
                                                    bool remote_accessible,
                                                    bool update_metadata) {
    int ret = transport_->registerLocalMemory(addr, length, "cpu", true, true);
    if (ret) {
        LOG(ERROR) << "rdma transport registerLocalMemory error, ret: "
                   << ret;
        return ret;
    }
    return 0;
}

int TcpTransportPerformance::unregisterLocalMemory(void *addr,
                                                      bool update_metadata) {
    int ret = transport_->unregisterLocalMemory(addr, true);
    if (ret) {
        LOG(ERROR) << "rdma transport unregisterLocalMemory error, ret: "
                   << ret;
        return ret;
    }
    return 0;
}

int TcpTransportPerformance::registerLocalMemoryBatch(
    const std::vector<TcpTransportPerformance::BufferEntry> &buffer_list,
    const std::string &location) {
    for (auto &buffer : buffer_list) {
        int ret = registerLocalMemory(buffer.addr, buffer.length, location,
                                      true, false);
        if (ret) {
            LOG(ERROR) << "TcpTransportPerformance registerLocalMemoryBatch "
                          "error, ret: "
                       << ret;
            return ret;
        }
    }
    return metadata_->updateLocalSegmentDesc();
}

int TcpTransportPerformance::unregisterLocalMemoryBatch(
    const std::vector<void *> &addr_list) {
    for (auto &addr : addr_list) {
        int ret = unregisterLocalMemory(addr, false);
        if (ret) {
            LOG(ERROR) << "TcpTransportPerformance "
                          "unregisterLocalMemoryBatch error, ret: "
                       << ret;
            return ret;
        }
    }
    return metadata_->updateLocalSegmentDesc();
}

Status TcpTransportPerformance::submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest> &entries) {
    if (entries.empty()) {
        LOG(ERROR)
            << "TcpTransportPerformance: empty transfer request batch";
        return Status::OK();
    }
    std::vector<TransferRequest> new_entries;
    new_entries.resize(entries.size());
    int index = 0;
    {
        std::lock_guard<std::mutex> lock(memcpy_mutex_);
        for (auto &request : entries) {
            if (offset_ + request.length > HUGE_HOST_SIZE) {
                offset_ = 0;
            }
            cudaError_t cuda_status =
                cudaMemcpy(static_cast<char *>(hostAddr_) + offset_, request.source,
                           request.length, cudaMemcpyDefault);
            if (cuda_status != cudaSuccess) {
                LOG(ERROR) << "TcpTransportPerformance: cudaMemcpy "
                              "error, ret: "
                           << cuda_status << ", hostAddr: " << hostAddr_
                           << ", offset_: " << offset_
                           << ", deviceAddr: " << request.source
                           << "len: " << request.length;
                return Status::InvalidArgument(
                    "TcpTransportPerformance: cudaMemcpy error");
            }
            new_entries[index] = request;
            new_entries[index].source = static_cast<char *>(hostAddr_) + offset_;
            offset_ += request.length;
        }
    }

    return transport_->submitTransfer(batch_id, new_entries);
}

Status TcpTransportPerformance::submitTransferTask(const std::vector<TransferTask *> &task_list) {
    if (task_list.empty()) {
        LOG(ERROR) << "TcpTransportPerformance: empty transfer task list";
        return Status::OK();
    }

    uint64_t total_length = 0;
    std::vector<TransferTask *> subTasks;
    uint64_t index = 0;
    int usedHugeDevNum = 0;
    {
        std::lock_guard<std::mutex> lock(memcpy_mutex_);
        while (index < task_list.size()) {
            std::unique_lock<std::mutex> lock_dev(dev_mtx_);
            dev_cv_.wait(lock_dev, [&] { return !mem_blocks[devId_]; });
            mem_blocks[devId_] = true;
            lock_dev.unlock();
            while (index < task_list.size()) {
                auto &task = *task_list[index];
                auto &request = *task.request;
                if (total_length + request.length > HUGE_DEVICE_SIZE) {
                    break;
                }
                cudaError_t cuda_status = cudaMemcpy(static_cast<char *>(hugeDevAddrs[devId_]) + total_length, request.source, request.length, cudaMemcpyDefault);
                if (cuda_status != cudaSuccess) {
                    LOG(ERROR)
                        << "TcpTransportPerformance: cudaMemcpy dtod error, ret: "
                        << cuda_status << ", offset_: " << offset_
                        << ", deviceAddr: " << request.source
                        << ", len: " << request.length;
                    return Status::InvalidArgument(
                        "TcpTransportPerformance: cudaMemcpy dtod "
                        "error");
                }
                subTasks.push_back(task_list[index]);
                ++index;
                total_length += request.length;
            }

            std::unique_lock<std::mutex> lock(transfer_mutex_);
            transferQueues_.emplace(std::move(subTasks), total_length, devId_);
            lock.unlock();
            transfer_cond_.notify_one();
            subTasks.clear();
            total_length = 0;
            devId_ = (devId_ + 1) & (HUGE_DEVICE_NUM - 1);
            usedHugeDevNum++;
        }
    }

    while (transfer_counter_.load() < usedHugeDevNum) {
        std::this_thread::yield();
    }

    transfer_counter_.fetch_sub(usedHugeDevNum);

    return Status::OK();
}

Status TcpTransportPerformance::getTransferStatus(BatchID batch_id,
                                                     size_t task_id,
                                                     TransferStatus &status) {
    return transport_->getTransferStatus(batch_id, task_id, status);
}

}  // namespace mooncake
