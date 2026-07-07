# CS149 Assignment 3 — 實驗結果紀錄

## 實驗環境

| 項目            | 內容                      |
| --------------- | ------------------------- |
| GPU             | NVIDIA GeForce RTX 4090 D |
| GPU 記憶體      | 24,111 MB                 |
| CUDA Capability | 8.9                       |
| Driver Version  | 560.35.03                 |
| CUDA Version    | 12.6                      |

---

## Part 1: SAXPY

### 測試參數

| 參數            | 值                                                    |
| --------------- | ----------------------------------------------------- |
| 陣列大小 N      | 100,000,000（1 億個元素）                             |
| alpha           | 2.0                                                   |
| threadsPerBlock | 512                                                   |
| 總資料量        | 3 × N × 4 bytes = **1.2 GB**（讀 x、讀 y、寫 result） |
| 計時次數        | 3 次                                                  |

### 成功執行結果（2026-07-06）

```
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce RTX 4090 D
   SMs:        114
   Global mem: 24111 MB
   CUDA Cap:   8.9
---------------------------------------------------------
Running 3 timing tests:
Effective BW by CUDA kernel only: 1.600 ms            [698.482 GB/s]
Effective BW by CUDA saxpy: 111.526 ms               [10.021 GB/s]
Effective BW by CUDA kernel only: 1.253 ms            [891.873 GB/s]
Effective BW by CUDA saxpy: 121.557 ms               [9.194 GB/s]
Effective BW by CUDA kernel only: 1.255 ms            [890.641 GB/s]
Effective BW by CUDA saxpy: 121.662 ms               [9.186 GB/s]
```

### 三次量測彙整

| Run      | Kernel only (ms) | Kernel only (GB/s) | Overall saxpy (ms) | Overall saxpy (GB/s) |
| -------- | ---------------- | ------------------ | ------------------ | -------------------- |
| 1        | 1.600            | 698.482            | 111.526            | 10.021               |
| 2        | 1.253            | 891.873            | 121.557            | 9.194                |
| 3        | 1.255            | 890.641            | 121.662            | 9.186                |
| **平均** | **1.369**        | **826.999**        | **118.248**        | **9.467**            |

### 觀察與分析（Writeup 草稿）

#### Q1: GPU vs CPU SAXPY 效能比較

- GPU kernel only 平均有效頻寬約 **827 GB/s**，執行時間約 **1.37 ms**。
- Assignment 1 的單執行緒 CPU SAXPY 通常在個位數到十幾 GB/s 量級。
- **結論**：GPU kernel 比 CPU 實作快約 **數十到上百倍**（以有效記憶體頻寬估算）。

#### Q2: Kernel only vs Overall 計時差異

| 計時範圍          | 平均時間  | 平均頻寬  | 瓶頸              |
| ----------------- | --------- | --------- | ----------------- |
| Kernel only       | 1.37 ms   | ~827 GB/s | GPU 記憶體頻寬    |
| Overall（含傳輸） | 118.25 ms | ~9.5 GB/s | CPU↔GPU PCIe 傳輸 |

- Kernel only 接近 RTX 4090 的理論記憶體頻寬（約 1008 GB/s），達到約 **82%** 利用率。
- Overall 需經過 `cudaMemcpy` 將 x、y 傳到 GPU，再將 result 傳回 CPU，共搬移 3 × 1.2 GB 資料，受 **PCIe 頻寬** 限制，頻寬降至約 9.5 GB/s。
- Overall 比 kernel only 慢約 **86 倍**，說明在資料需來回 CPU/GPU 時，傳輸成本遠大於計算本身。
- 作業 README 以 AWS T4（~320 GB/s）為參考；本實驗使用 RTX 4090 D，GPU 記憶體頻寬更高，故 kernel only 結果也更高。

---

## Part 2: Parallel Prefix-Sum (Scan)

> 待完成

---

## Part 3: Circle Renderer

> 待完成
