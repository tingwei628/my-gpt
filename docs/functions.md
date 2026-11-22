# 從命令列到文字輸出的函式圖解

先讀 [miniGPT 與推論引擎](../README.md)，再沿這份文件追蹤「載入一次、生成一次、評分一次」。[模型函式圖解](model-functions.md) 接著說明 GPU 計算。

每張圖把相關函式放在一起，每一列說明一個具名函式的輸入、動作與輸出。建構式負責建立物件，Dispose 負責釋放資源。區域函式會標出它所屬的外層函式。record 是保存結果或設定的資料型別。

## 第 0、1 課如何走進完整專案

第 0 課有兩個手寫函式。第 1 課則直接由檔案頂層敘述執行兩段練習。下面的區塊名稱對應程式中的運算段落。

```mermaid
flowchart TD
    I["兩個陣列 1,2 與 3,4"] --> M["第 0 課 Program.Main<br/>選 CUDA，配置記憶體，複製輸入"]
    M --> A["Program.Add<br/>一個 index 算一個加法"]
    A --> O["Main 同步、讀回、逐項驗算<br/>4,6"]
    W["第 1 課頂層敘述<br/>x=2，目標=4，權重=1"] --> G["預測、誤差、梯度<br/>權重變為 1.4"]
    T["頂層敘述的 bigram 段<br/>abcab 與 3×3 分數表"] --> P["softmax 與交叉熵梯度<br/>逐組更新分數"]
    P --> N["印出 a→b、b→c、c→a"]
```

| 函式或區塊 | 輸入 | 做的事 | 輸出或狀態 |
|---|---|---|---|
| 第 0 課 Program.Main | 程式內兩組整數 | 選裝置、編譯 Add kernel、傳資料、執行、同步與驗算 | 結果、結束碼 0 或 1 |
| 第 0 課 Program.Add | index 與三個 GPU ArrayView | 把相同位置的兩數相加 | 寫入 result[index] |
| 第 1 課單權重頂層敘述 | 輸入、目標、權重 | 用半平方誤差求梯度，更新一次 | 新權重及更新前後 loss |
| 第 1 課 bigram 頂層敘述 | 原文與分數表 | 反覆取相鄰字，更新下一字的候選分數 | 三個字各自最可能接的下一字 |

對照 [CUDA 程式](../lessons/00-cuda/Program.cs) 與 [學習程式](../lessons/01-learning/Program.cs)。

## Program.cs 決定這次要做什麼

```mermaid
flowchart TD
    A["minigpt.cmd 轉交參數"] --> B["頂層 try / catch<br/>呼叫 Program.Run"]
    B --> C{"Program.Run<br/>讀取命令"}
    C -->|"空參數或 chat"| D["InteractiveCli.Run"]
    C -->|"generate / score / benchmark"| E["InferenceCommands.Run"]
    C -->|"math / attention"| F["Model.ShowLinear<br/>或 Evaluate 後 ShowBlock"]
    C -->|"check"| G["Checks.Run"]
    C -->|"train / resume"| H["讀語料、準備視窗<br/>Model.Train 反覆更新"]
    H --> V["區域函式 Validation<br/>驗證段的平均 loss"]
    H --> S["Checkpoint.Save"]
    S --> P["Program.Generate<br/>印出訓練後樣本"]
```

| 函式或區塊 | 輸入 | 做的事 | 輸出或狀態 |
|---|---|---|---|
| 頂層入口 | 命令列 args | 呼叫 Run，將檔案錯誤轉為一般提示 | 程式結束碼 |
| Program.Run | args | 分派命令，訓練時讀資料、建 batch、執行更新與存檔 | 成功 0、無效命令 1，或交給入口處理錯誤 |
| Run 內的 Validation | 外層的 model、驗證段、T 與 batch | 取完整驗證視窗，重複填滿 batch，呼叫 Evaluate | 各完整視窗平均 loss |
| Program.Generate | 已在訓練中使用的 Model、字表、開頭、長度 | 保留最近 T 字元，呼叫 Forward，逐次取最大分數 | 包含開頭的樣本文字 |

Program.Generate 供訓練結束後列印樣本使用，與下面的 InferenceSession.Generate 都沿用模型前向計算。前者使用訓練中的 model，後者管理專門載入的推論 model。[程式入口](../lessons/miniGPT/Program.cs)

## 互動介面如何一直等待下一次輸入

```mermaid
flowchart TD
    A["InteractiveCli.Run<br/>args"] --> B{"有指定模型嗎"}
    B -->|"有"| C["使用指定路徑"]
    B -->|"無"| D["DefaultCheckpoint<br/>尋找專案內唯一 ckpt"]
    C --> L["new InferenceSession<br/>載入一次"]
    D --> L
    L --> R["讀取一行"]
    R --> Q{"Run 判斷輸入"}
    Q -->|"一般文字"| G["session.Generate<br/>onCharacter 回呼逐字顯示"]
    Q -->|"/length /info /help"| S["調整設定或顯示資訊"]
    G --> R
    S --> R
    Q -->|"/exit 或 EOF"| X["session.Dispose<br/>釋放模型"]
```

| 函式 | 輸入 | 做的事 | 輸出或狀態 |
|---|---|---|---|
| InteractiveCli.Run | 可選 checkpoint 路徑 | 建立 session、讀取輸入、解析斜線指令，將新字元寫到 Console | 連續續寫與結束碼 0 |
| InteractiveCli.DefaultCheckpoint | 執行目錄 | 向上找 MiniGPT.csproj，確認該專案內恰有一個 ckpt | 模型路徑，零檔或多檔時給提示 |

空字串會回到輸入提示，單一空白則能作為模型輸入。字表外文字顯示錯誤後，可以繼續輸入。onCharacter 是傳給 Generate 的回呼，每產生一個新字元，就執行 Console.Write。[互動介面](../lessons/miniGPT/InteractiveCli.cs)

## 單次指令如何共用同一套推論程式

每次命令都建立自己的 session。benchmark 在一次執行中，會讓四個開頭共用同一個已載入的 session。

```mermaid
flowchart TD
    A["InferenceCommands.Run<br/>命令與參數"] --> B{"是哪種命令"}
    B -->|"generate"| C["Count<br/>讀取長度或預設 100"]
    C --> G["new InferenceSession<br/>Generate → 印出文字"]
    B -->|"benchmark"| D["Count<br/>要求長度大於 0"]
    D --> E["載入一次，暖機一次<br/>固定四個開頭 Generate"]
    E --> F["Run 的統計段<br/>計時、句型與重複句"]
    B -->|"score"| SC["new InferenceSession"]
    SC --> S["讀取文本，統一 LF<br/>session.Score → 印出指標"]
    G --> X["session.Dispose"]
    F --> X
    S --> X
```

| 函式 | 輸入 | 做的事 | 輸出或狀態 |
|---|---|---|---|
| InferenceCommands.Run | generate、benchmark 或 score 參數 | 建立 session、執行要求並顯示結果 | 結束碼 0，參數錯誤時拋出提示 |
| InferenceCommands.Count | args 與長度參數的位置 | 省略時回傳 100，有提供時解析非負整數 | 新增字元數 |

benchmark 的句型檢查與計時寫在 Run 的命令分支中。它使用教材英文提示詞，對每個開頭各自計算完整句與重複句。[單次命令](../lessons/miniGPT/InferenceCommands.cs)

## InferenceSession 如何載入、編碼與生成

```mermaid
flowchart TD
    P["checkpoint 路徑"] --> C["InferenceSession 建構式<br/>Checkpoint.Load(false)"]
    C --> Y["Model.Sync<br/>完成載入與量測"]
    Y --> S["可重用的 session"]
    S --> G["Generate<br/>開頭與新增長度"]
    G --> E["Encode<br/>文字查字表"]
    E --> Q{"還要生成字元嗎"}
    Q -->|"是"| W["Generate 的視窗段<br/>最近 T 字元，補到 T"]
    W --> F["Model.Forward<br/>取得各位置 logits"]
    F --> N["Generate 的選字段<br/>讀最後有效列的最大分數"]
    N --> O["加入文字與編號<br/>執行 onCharacter"]
    O --> Q
    Q -->|"否，包含 count=0"| R["Generation<br/>文字與耗時統計"]
    S --> D["Dispose → Model.Dispose"]
```

| 函式 | 輸入 | 做的事 | 輸出或狀態 |
|---|---|---|---|
| InferenceSession 建構式 | checkpoint 路徑 | 載入推論模式 model 與字表，同步並記錄時間 | session，可讀取 Device、Step、LoadMs 等資訊 |
| Encode | 文字 | 把每個 UTF-16 字元查成字表編號，遇到字表外字元時提示 | int[] |
| Generate | prompt、count、可選 onCharacter | 驗證輸入、逐次前向計算與貪婪選字、記錄首字及整段時間 | Generation，文字含原開頭 |
| Dispose | 已載入的 session | 呼叫 model.Dispose | 釋放 GPU 資源 |

count=0 時回傳原開頭。若開頭超過 T，先驗證全部字元，再只取最近 T 字元計算。補齊的位置不影響先前有效位置，這由模型的因果遮罩保證。[推論 session](../lessons/miniGPT/InferenceSession.cs)

## Score 如何把前向分數變成評估指標

```mermaid
flowchart TD
    T["有已知下一字答案的文本"] --> C["Score 入口"]
    C --> E["Encode<br/>取得字元編號"]
    E --> S["Score 的分塊段<br/>每次取至多 T 個輸入位置"]
    S --> F["Model.Forward<br/>預測各位置"]
    F --> L["Score 的計分段<br/>對照原文下一字<br/>累加交叉熵與命中數"]
    L --> Q{"還有目標位置嗎"}
    Q -->|"有"| S
    Q -->|"無"| O["TextScore<br/>N-1 次預測的平均 loss 與正確率"]
    O --> P["Perplexity 屬性<br/>exp 平均 loss"]
```

| 函式或屬性 | 輸入 | 做的事 | 輸出 |
|---|---|---|---|
| InferenceSession.Score | 至少兩字的文本 | 每 T 位置重置上下文，逐位置計分，尾段也計入，求平均 | TextScore |
| TextScore.Perplexity | TextScore.Loss | 計算 Math.Exp(Loss) | 字元 perplexity |

例如 abcab 有四個下一字元目標 b、c、a、b。使用較小的 T=3 時，前三個目標放第一段，最後一個放尾段，每個目標恰好評一次。這裡的已知文本評分和逐字生成使用不同的上下文切法。

Generation 保存 Text、NewCharacters、FirstCharacterMs、CharactersPerSecond。TextScore 保存 Predictions、Loss、Accuracy。這兩個 record 的建構式由 C# 依欄位自動產生，運算邏輯集中在上表。

## Checkpoint 如何連接訓練與推論

```mermaid
flowchart TD
    M["訓練模式 model、字表、資料指紋"] --> S["Save<br/>寫入格式、尺寸、步數、權重與動量"]
    S --> T["暫存檔關閉後<br/>替換目標 ckpt，清除暫存"]
    T --> L["Load<br/>讀取與驗證格式、尺寸、字表"]
    L --> B{"forTraining"}
    B -->|"true"| A["建立訓練 Model<br/>還原權重和 M、V"]
    B -->|"false"| I["建立 batch 1 推論 Model<br/>讀權重，略過檔案中的 M、V"]
    A --> R["回傳 Model<br/>另帶出字表與資料指紋"]
    I --> R
```

| 函式 | 輸入 | 做的事 | 輸出或狀態 |
|---|---|---|---|
| Checkpoint.Save | path、訓練 model、alphabet、dataHash、overwrite | 寫完整暫存檔，再換到目標路徑，finally 清除暫存 | ckpt 檔案 |
| Checkpoint.Load | path、forTraining | 檢查 MiniGPT/1，依參數順序還原資料，檢查檔案完整性 | Model，以及 out 字表、語料指紋 |

Save 的 overwrite 預設為 false。Load 失敗時會 Dispose 已建立的 model，再回報錯誤。M 是梯度移動平均，V 是梯度平方移動平均，保存它們可以接續 AdamW 更新。[模型存檔](../lessons/miniGPT/Checkpoint.cs)

## Checks 如何核對模型

```mermaid
flowchart TD
    A["Checks.Run<br/>建立兩段小序列與隨機權重"] --> G["Model.Evaluate / Differentiate<br/>GPU 分數、loss 與梯度"]
    A --> C["Reference.Predict<br/>獨立 CPU 分數"]
    C --> L["Run 內的 CpuLoss<br/>對已知目標算平均交叉熵"]
    L --> F["改動抽樣權重 ±ε<br/>再求 CpuLoss"]
    G --> R["Require<br/>比對差距與預期條件"]
    F --> R
    A --> V["Run 其他檢查段<br/>因果、batch、固定批次學習<br/>推論、Score、checkpoint 還原"]
    V --> R
    R --> O["通過則繼續<br/>失敗則拋出檢查訊息"]
```

| 函式 | 輸入 | 做的事 | 輸出或狀態 |
|---|---|---|---|
| Checks.Run | 程式內設定的小模型與例子 | 比較 CPU/GPU 分數和 loss，從每組參數抽一個元素驗梯度，再做遮罩、學習與還原檢查 | 通過訊息，或中斷於失敗項目 |
| Checks.Require | condition、message | 判斷條件是否成立 | 成立則返回，失敗則拋例外 |
| Run 內的 CpuLoss | 外層目前權重、input、target | 呼叫 Reference.Predict，再用 double 計算平均交叉熵 | 一個可與 GPU 對照的 loss |

CpuLoss 會讀到外層被暫時加減的權重，因此能用兩次獨立 loss 估計該元素的斜率。檢查結束前會把該元素還原。[檢查程式](../lessons/miniGPT/Checks.cs)

接著閱讀 [Model、Kernels 與 Reference 的函式圖解](model-functions.md)，把 Model.Forward 展開到每個 GPU 運算。
