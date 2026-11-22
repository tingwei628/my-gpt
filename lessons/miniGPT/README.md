# 從矩陣乘法到可續訓的 miniGPT

在專案根目錄依序執行以下各課。硬體與 .NET 設定沿用[第 0 課](../../README.md)。每一個指令都是一次獨立的程式執行。

## 第 2 課　把分數算出來

執行 `dotnet run --project lessons/miniGPT -- math`。假設兩個位置的輸入分別是 [1,2] 和 [3,4]，第一個候選字的權重是 [2,1]，第二個是 [0,3]。第一個位置給兩個候選字的分數就是 1×2+2×1=4、1×0+2×3=6。先自己算第二個位置，再對照 GPU 印出的四個分數 [4,6,10,12]。到這裡，每個分數都還可大於 1，稱為 **logit**。

回想上一課的 softmax。假如三個候選字的機率是 [0.1,0.6,0.3]，正解是第二個字，交叉熵就是 −ln(0.6) ≈ 0.511。若模型把正解的機率提高到 0.9，loss 會變為多少？手算可用 −ln(0.9) ≈ 0.105 核對。[GPU 算子](Kernels.cs) 的 Linear 算分數，Loss 用等價、數值較穩定的公式計算每個位置的 loss。

## 第 3 課　看見每個字可以讀到哪裡

執行 `dotnet run --project lessons/miniGPT -- attention`。輸入是 abc，以 0、1、2 表示。每個字有一列 8 個數字，稱為嵌入向量。先將字元嵌入和位置嵌入相加，再從同一列算出三組向量：Q 表示這個位置要尋找的資訊，K 用來和 Q 算相似度，V 是可讀取的內容。

這個練習把 8 個數字拆給兩個頭，每個頭 4 個。用 Q 與各個 K 的內積除以 √4 得到分數，把未來位置排除後做 softmax，再依這些機率加權平均 V。程式會印出第一個頭的 3×3 機率表。列代表目前字，欄代表要讀取的字。第一列應是 [1,0,0]，第二列的 c 欄應為 0。找出第二列必定為零的位置，再檢查第三列的總和是否接近 1（輸出取到小數第三位）。

跟著 [Model 建構式](Model.cs) 讀一個 block。Norm 把每列數字調到相近尺度，Gain、Bias 讓模型學習合適的縮放與位移。Scores → Softmax → Mix 完成注意力；Add 把得到的新資訊加回原輸入。接著先擴寬每列、經過平滑的非線性函式 GELU，再縮回原寬並加回去。這條捷徑讓訊息與梯度都能穿過整個 block。最後 Linear 為每個位置產生字表大小的候選分數。[minGPT 的對照實作](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py)

## 第 4 課　證明模型學得動

執行 `dotnet run --project lessons/miniGPT -- check`。這一課用兩組各四個位置的小序列。先看 [CPU 參考答案](Reference.cs) 的二維陣列，再看 [數值檢查](Checks.cs) 怎麼用同一組權重比較 GPU 分數和 loss。

1. 前向計算依序產生分數與 loss。反向傳播沿相反順序，逐層算出「輸入稍微增加時，loss 會變多少」。同一個字出現兩次時，兩個位置都會對它的嵌入權重貢獻梯度。
2. 檢查器在每組參數選一個元素，先把權重加一點，再減一點。兩次 CPU loss 的差除以權重的差，是獨立的斜率估計，應接近 GPU 算出的梯度。
3. 改動第一段的末字，前面三個位置及另一段的分數都應相同。固定同一批資料更新 100 步，最後 loss 應低於起點的 35%。AdamW 用梯度的移動平均與平方的移動平均決定更新幅度，另外對指定權重施加衰減。
4. 存檔再讀回來，確認輸出逐項相同。兩個模型再各更新一步，比較所有權重，檢查動量是否也有正確還原。通過後最後一行是「checkpoint 還原 logits 及下一步更新一致」。

## 第 5 課　訓練自己的模型、關掉程式、繼續生成

在專案根目錄使用一個新的檔名。以下指令共訓練 125 步。

```powershell
dotnet run --project lessons/miniGPT -- train lessons/miniGPT/my-first.ckpt 100
dotnet run --project lessons/miniGPT -- generate lessons/miniGPT/my-first.ckpt "the "
dotnet run --project lessons/miniGPT -- resume lessons/miniGPT/my-first.ckpt 25
```

第一條指令配置 2 層、4 個頭、嵌入寬度 64 的模型。每步讀 1 段 64 字元，每個位置的答案是原文中的下一字。例如原文 abcd，輸入 abc，答案 bcd。GPU buffers 表示程式配置的張量空間，驅動程式還會使用其他顯存。暖機後的 tokens/s 是每秒處理多少個訓練位置。比較開頭與結尾的驗證 loss，再看模型能否續寫出語料裡的單字。教材語料由簡單英文句型組成，適合觀察模型學會拼字及句型的過程。

第二條指令重新讀取檔案，從開頭文字逐次挑分數最高的字；超過 64 個位置時只讀最近 64 個。這種選字法稱為貪婪解碼，同一個 checkpoint 與開頭會得到相同文字。試著把開頭換成 `"the fox "`，比較兩次續寫。第三條指令應印出 step 101 到 step 125 的進度。它會接著使用儲存的權重、AdamW 狀態、步數及資料順序。

可以把自己的 UTF-8 純文字檔路徑放在步數後，例如 `dotnet run --project lessons/miniGPT -- train lessons/miniGPT/custom.ckpt 100 my-text.txt 4`，最後的 4 是每批段數。續訓時也要提供同一份文本，例如 `dotnet run --project lessons/miniGPT -- resume lessons/miniGPT/custom.ckpt 25 my-text.txt`。教材字表支援 2 至 1024 個單一 UTF-16 單元的字元，包含常見中文字；含 emoji 或擴展漢字等代理對的語料會在訓練前提示。資料依原順序切分，約前九成用來抽訓練視窗、後一成驗證。兩段各需至少 65 字元，驗證用字必須曾在訓練段出現。內建[語料](corpus.txt)保留未出現在訓練段的完整句子組合。

## 筆電實測

**練習：比較學習率。** 學習率控制每次更新權重的步幅。從同一個模型複製兩份，各跑相同步數，才能把差異歸因到步幅。以下以第 5 課建立的檔案為起點，請使用尚未存在的副本名稱：

```powershell
[IO.File]::Copy("lessons/miniGPT/my-first.ckpt", "lessons/miniGPT/trial-small.ckpt")
dotnet run --project lessons/miniGPT -- resume lessons/miniGPT/trial-small.ckpt 100 --lr 0.0003
```

另做一份副本，把學習率改成 0.003，再比較兩次的「訓練後驗證 loss」及同一句開頭的生成結果。`--lr` 只套用於本次指令，省略時使用預設值 0.003。要接著用較小步幅訓練，每次續訓都要指定相同的 `--lr`。

在 GTX 1650 Max-Q 4 GB 的一次實測中，固定預設語料與種子，每批 4 段訓練 500 步，驗證 loss 從 3.1469 降至 0.2053，暖機後約 32,849 tokens/s。模型明確配置了 12.18 MiB 緩衝區。另一次相同配置跑 300 步，以 nvidia-smi 取得 20 次整卡使用量，最高讀值為 63 MiB；取樣間隔內的瞬間用量可能更高。每批 1 段時，連跑 125 步與先跑 100 步、重新啟動後再跑 25 步的 checkpoint，SHA-256 相同。

## 對照四部影片

[神經網路與梯度](https://www.youtube.com/watch?v=SXnHqFGLNxA) 對應第 1 課的單權重。[預訓練](https://www.youtube.com/watch?v=EFXiQSxa4d8) 對應 bigram 與第 5 課的答案位移。[PyTorch 張量](https://www.youtube.com/watch?v=r1bquDz5GGA) 可和第 2、4 課的矩陣與梯度對讀。[Transformer](https://www.youtube.com/watch?v=CfJ3Cxtlcps) 對應第 3 課印出的注意力表及整個 block。
