using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using ILGPU.Runtime;
using MiniGPT;

try { return Run(args); }
catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
{
    Console.Error.WriteLine("檔案操作失敗。請確認模型或語料存在、路徑與存取權限正確，且檔案可供讀寫。");
    return 1;
}
catch (Exception ex) { Console.Error.WriteLine(ex.Message); return 1; }

static int Run(string[] args)
{
    if (args is ["--help"] or ["-h"])
    {
        Console.WriteLine("miniGPT 互動命令列");
        Console.WriteLine("不帶參數啟動，或 chat [模型.ckpt]。進入後輸入文字，按 Enter 續寫。");
        Console.WriteLine("generate 模型.ckpt 開頭 [新增字元數] | benchmark 模型.ckpt [新增字元數] | score 模型.ckpt 文本.txt");
        return 0;
    }
    if (args.Length == 0 || args[0] == "chat") return InteractiveCli.Run(args);
    if (args is ["check"]) { Checks.Run(); return 0; }
    if (args is ["math"] or ["attention"])
    {
        using var demo = new Model(new(1, 3, 8, 2, 1), 3);
        if (args[0] == "math") demo.ShowLinear();
        else { demo.Evaluate([0,1,2], [1,2,0]); demo.ShowBlock(); }
        return 0;
    }
    if (args.Length > 0 && args[0] is "generate" or "benchmark" or "score")
        return InferenceCommands.Run(args);
    float learningRate = Model.DefaultLearningRate;
    if (args.Length >= 5 && args[^2] == "--lr")
    {
        if (!float.TryParse(args[^1], NumberStyles.Float, CultureInfo.InvariantCulture, out learningRate)
            || !float.IsFinite(learningRate) || learningRate <= 0)
            throw new ArgumentException("--lr 需要大於零的有限數值，例如 0.0003。");
        args = args[..^2];
    }
    if (args.Length < 3 || args.Length > 5 || args[0] is not ("train" or "resume") ||
        !int.TryParse(args[2], out int steps) || steps < 1)
    {
        Console.Error.WriteLine("用法：math | attention | check | train 新檔.ckpt 步數 [語料.txt [batch]] [--lr 學習率] | resume 檔.ckpt 步數 [語料.txt] [--lr 學習率] | generate 檔.ckpt 開頭 [新增字元數] | benchmark 檔.ckpt [新增字元數] | score 檔.ckpt 文本.txt");
        return 1;
    }
    string path = args[1];
    bool fresh = args[0] == "train";
    using var trainingLock = new FileStream(path + ".lock", FileMode.OpenOrCreate,
        FileAccess.ReadWrite, FileShare.None, 1, FileOptions.DeleteOnClose);
    if (fresh && File.Exists(path)) throw new IOException("檔案已存在，請用 resume 繼續訓練或指定新檔名。");
    int batch = 1;
    if (args.Length == 5 && (!fresh || !int.TryParse(args[4], out batch) || batch is < 1 or > 4))
        throw new ArgumentException("新模型 batch 可設為 1 至 4，續訓沿用原設定。");
    string text = File.ReadAllText(args.Length >= 4 ? args[3] : Path.Combine(AppContext.BaseDirectory, "corpus.txt")).ReplaceLineEndings("\n");
    int cut = text.Length * 9 / 10;
    if (text.Any(char.IsSurrogate)) throw new ArgumentException("教材字表目前支援單一 UTF-16 單元的字元，請移除 emoji 或其他代理對。");
    int newline = text.IndexOf('\n', cut);
    if (newline >= 0 && text.Length - newline - 1 >= 65) cut = newline + 1;
    string alphabet = new(text[..cut].Distinct().Order().ToArray());
    if (alphabet.Length is < 2 or > 1024) throw new ArgumentException("教材支援 2 至 1024 個字元。");
    string hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text)));
    string savedAlphabet = alphabet, savedHash = hash;
    using var model = fresh ? new Model(new Config(Batch: batch), alphabet.Length) : Checkpoint.Load(path, out savedAlphabet, out savedHash);
    if (alphabet != savedAlphabet || hash != savedHash) throw new InvalidDataException("續訓需要與原訓練相同的語料。");
    int[] ids = text.Select(c => alphabet.IndexOf(c)).ToArray();
    if (ids.Contains(-1)) throw new InvalidDataException("驗證段含有訓練段未出現的字元。");
    int t = model.Config.T;
    if (cut <= t || ids.Length - cut <= t) throw new InvalidDataException($"訓練段及後 10% 驗證段各需至少 {t + 1} 個字元。");
    Console.WriteLine($"CUDA {model.Device} | 模型配置的 GPU buffers {model.BufferBytes / 1048576.0:F2} MiB");
    Console.WriteLine($"學習率 {learningRate}，訓練前驗證 loss {Validation():F4}");
    var watch = Stopwatch.StartNew();
    for (int i = 0; i < steps; i++)
    {
        // 固定每一步選到的視窗，讓重新載入後也能接上相同的資料順序。
        var rng = new Random(model.Step);
        var input = new int[model.Config.Batch * t];
        var labels = new int[input.Length];
        for (int b = 0; b < model.Config.Batch; b++)
        {
            int start = rng.Next(cut - t);
            Array.Copy(ids, start, input, b * t, t);
            Array.Copy(ids, start + 1, labels, b * t, t);
        }
        float loss = model.Train(input, labels, learningRate);
        if (i == 0 || (i + 1) % 25 == 0 || i == steps - 1)
            Console.WriteLine($"step {model.Step}  train loss {loss:F4}");
        if (i == 0) { model.Sync(); watch.Restart(); }
    }
    model.Sync();
    watch.Stop();
    if (steps > 1) Console.WriteLine($"暖機後 {(steps - 1) * (double)t * model.Config.Batch / watch.Elapsed.TotalSeconds:F0} tokens/s");
    Console.WriteLine($"訓練後驗證 loss {Validation():F4}");
    Checkpoint.Save(path, model, alphabet, hash, overwrite: !fresh);
    Console.WriteLine(Generate(model, alphabet, text[..3], 60));
    return 0;

    float Validation()
    {
        float sum = 0;
        int count = 0;
        for (int i = cut; i + t < ids.Length; i += t)
        {
            // 重複同一段填滿 batch，各段的平均 loss 等於單段的 loss。
            var x = Enumerable.Range(0, model.Config.Batch * t).Select(j => ids[i + j % t]).ToArray();
            var y = Enumerable.Range(0, x.Length).Select(j => ids[i + j % t + 1]).ToArray();
            sum += model.Evaluate(x, y);
            count++;
        }
        return sum / count;
    }
}

static string Generate(Model model, string alphabet, string prompt, int count)
{
    if (prompt.Length == 0 || prompt.Any(c => !alphabet.Contains(c))) throw new ArgumentException("開頭需包含至少一個字，且每個字都在訓練字表內。");
    string text = prompt;
    int t = model.Config.T;
    for (int i = 0; i < count; i++)
    {
        string prefix = text.Length > t ? text[^t..] : text;
        var input = new int[t * model.Config.Batch];
        for (int j = 0; j < prefix.Length; j++) input[j] = alphabet.IndexOf(prefix[j]);
        var logits = model.Forward(input);
        int offset = (prefix.Length - 1) * alphabet.Length;
        int next = Enumerable.Range(0, alphabet.Length).MaxBy(k => logits[offset + k]);
        text += alphabet[next];
    }
    return text;
}
