using System.Diagnostics;

namespace MiniGPT;

internal sealed record Generation(string Text, int NewCharacters, double FirstCharacterMs, double CharactersPerSecond);
internal sealed record TextScore(int Predictions, double Loss, double Accuracy)
{
    public double Perplexity => Math.Exp(Loss);
}

// 一個 session 載入一次權重，依序處理多個要求。相同 session 的呼叫應逐一執行。
internal sealed class InferenceSession : IDisposable
{
    private readonly Model model;
    public string Alphabet { get; }
    public string Device => model.Device;
    public int Step => model.Step;
    public long BufferBytes => model.BufferBytes;
    public double LoadMs { get; }

    public InferenceSession(string checkpoint)
    {
        var timer = Stopwatch.StartNew();
        model = Checkpoint.Load(checkpoint, out string alphabet, out _, forTraining: false);
        Alphabet = alphabet;
        model.Sync();
        LoadMs = timer.Elapsed.TotalMilliseconds;
    }

    private int[] Encode(string text)
    {
        var ids = new int[text.Length];
        for (int i = 0; i < text.Length; i++)
        {
            ids[i] = Alphabet.IndexOf(text[i]);
            if (ids[i] < 0) throw new ArgumentException($"第 {i + 1} 個字元不在訓練字表內。");
        }
        return ids;
    }

    public Generation Generate(string prompt, int count = 100, Action<char>? onCharacter = null)
    {
        if (prompt.Length == 0) throw new ArgumentException("開頭至少需要一個字元。");
        if (count < 0) throw new ArgumentException("新增字元數需大於或等於零。");
        var encoded = Encode(prompt).ToList();
        int t = model.Config.T, vocab = Alphabet.Length;
        var input = new int[t];
        var result = new System.Text.StringBuilder(prompt);
        var timer = Stopwatch.StartNew();
        double firstMs = 0;
        for (int i = 0; i < count; i++)
        {
            int length = Math.Min(t, encoded.Count);
            Array.Clear(input);
            encoded.CopyTo(encoded.Count - length, input, 0, length);
            float[] logits = model.Forward(input);
            int offset = (length - 1) * vocab;
            int next = 0;
            for (int j = 1; j < vocab; j++)
                if (logits[offset + j] > logits[offset + next]) next = j;
            encoded.Add(next);
            result.Append(Alphabet[next]);
            onCharacter?.Invoke(Alphabet[next]);
            if (i == 0) firstMs = timer.Elapsed.TotalMilliseconds;
        }
        timer.Stop();
        return new(result.ToString(), count, firstMs,
            count == 0 ? 0 : count / timer.Elapsed.TotalSeconds);
    }

    public TextScore Score(string text)
    {
        int[] ids = Encode(text);
        if (ids.Length < 2) throw new ArgumentException("評分文本至少需要兩個字元。");
        int t = model.Config.T, vocab = Alphabet.Length, correct = 0;
        double loss = 0;
        var input = new int[t];
        // 每個下一字元目標只評一次。每 T 個位置重置上下文，尾段也計入。
        for (int start = 0; start < ids.Length - 1; start += t)
        {
            int length = Math.Min(t, ids.Length - 1 - start);
            Array.Clear(input);
            Array.Copy(ids, start, input, 0, length);
            float[] logits = model.Forward(input);
            for (int row = 0; row < length; row++)
            {
                int offset = row * vocab, best = 0;
                for (int j = 1; j < vocab; j++)
                    if (logits[offset + j] > logits[offset + best]) best = j;
                double peak = logits[offset + best], total = 0;
                for (int j = 0; j < vocab; j++) total += Math.Exp(logits[offset + j] - peak);
                int target = ids[start + row + 1];
                loss += peak - logits[offset + target] + Math.Log(total);
                if (best == target) correct++;
            }
        }
        return new(ids.Length - 1, loss / (ids.Length - 1), (double)correct / (ids.Length - 1));
    }

    public void Dispose() => model.Dispose();
}
