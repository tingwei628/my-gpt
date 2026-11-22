using ILGPU.Runtime;

namespace MyGpt;

internal static class Checks
{
    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }

    public static void Run()
    {
        var c = new Config(2, 4, 8, 2, 1);
        using var m = new Model(c, 3);
        int[] input = [0,1,0,2, 2,0,1,0], target = [1,0,2,0, 0,1,0,2];
        var p = m.Parameters.ToDictionary(x => x.Name, x => x.Tensor.Value.GetAsArray1D());
        float initial = m.Evaluate(input, target);
        var expected = Reference.Predict(c, 3, p, input);
        var logits = m.Logits.Value.GetAsArray1D();
        Require(expected.Cast<double>().Zip(logits, (a,b) => Math.Abs(a-b)).Max() < 2e-5, "CPU/GPU logits 不一致");
        Require(Math.Abs(initial - CpuLoss()) < 2e-5, "交叉熵不一致");
        m.Differentiate();
        foreach (var param in m.Parameters)
        {
            var analytic = param.Tensor.Grad!.GetAsArray1D();
            int index = Enumerable.Range(0, analytic.Length).MaxBy(i => Math.Abs(analytic[i]));
            var values = p[param.Name];
            float old = values[index];
            values[index] = old + 0.0001f;
            double high = CpuLoss();
            values[index] = old - 0.0001f;
            double low = CpuLoss();
            double numerical = (high - low) / ((old + 0.0001f) - values[index]);
            values[index] = old;
            Require(Math.Abs(numerical - analytic[index]) < 1e-5 + Math.Abs(numerical) * 0.002, $"梯度不符 {param.Name} {numerical} {analytic[index]}");
        }
        Console.WriteLine($"CPU 與 GPU logits、loss 及 {p.Count} 組參數的抽樣梯度吻合");
        int[] changed = (int[])input.Clone();
        changed[3] = 1;
        m.Evaluate(changed, target);
        var altered = m.Logits.Value.GetAsArray1D();
        Require(logits.Where((_,i) => i / 3 != 3).SequenceEqual(altered.Where((_,i) => i / 3 != 3)), "因果或批次隔離失敗");
        for (int step = 0; step < 100; step++) m.Train(input, target);
        float final = m.Evaluate(input, target);
        Require(final < initial * 0.35, $"固定批次未學起來 {final}");
        Console.WriteLine($"因果遮罩通過，固定批次 100 步 loss {initial:F4} → {final:F4}");
        string file = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".ckpt");
        try
        {
            Checkpoint.Save(file, m, "abc", "test");
            using var copy = Checkpoint.Load(file, out string alphabet, out _);
            Require(alphabet == "abc" && copy.Step == m.Step, "還原設定失敗");
            copy.Evaluate(input, target);
            Require(copy.Logits.Value.GetAsArray1D().SequenceEqual(m.Logits.Value.GetAsArray1D()), "還原 logits 失敗");
            using (var inference = new InferenceSession(file))
            {
                Require(inference.BufferBytes < m.BufferBytes, "推論未減少 GPU 緩衝區");
                var weights = m.Parameters.ToDictionary(x => x.Name, x => x.Tensor.Value.GetAsArray1D());
                const string text = "abcabca";
                double expectedLoss = 0;
                int expectedCorrect = 0;
                for (int start = 0; start < text.Length - 1; start += c.T)
                {
                    int length = Math.Min(c.T, text.Length - 1 - start);
                    var window = new int[c.T];
                    for (int j = 0; j < length; j++) window[j] = "abc".IndexOf(text[start + j]);
                    var cpu = Reference.Predict(c with { Batch = 1 }, 3, weights, window);
                    for (int row = 0; row < length; row++)
                    {
                        int answer = "abc".IndexOf(text[start + row + 1]);
                        int best = Enumerable.Range(0, 3).MaxBy(i => cpu[row,i]);
                        double peak = cpu[row,best];
                        expectedLoss += peak - cpu[row,answer] + Math.Log(Enumerable.Range(0, 3).Sum(i => Math.Exp(cpu[row,i] - peak)));
                        if (best == answer) expectedCorrect++;
                    }
                }
                var score = inference.Score(text);
                Require(score.Predictions == 6 && Math.Abs(score.Loss - expectedLoss / 6) < 2e-5
                    && Math.Abs(score.Accuracy - expectedCorrect / 6.0) < 1e-10, "推論文本評分或尾段不符 CPU");
                string generated = "ab";
                for (int j = 0; j < 8; j++)
                {
                    string prefix = generated.Length > c.T ? generated[^c.T..] : generated;
                    var window = new int[c.Batch * c.T];
                    for (int k = 0; k < prefix.Length; k++) window[k] = "abc".IndexOf(prefix[k]);
                    var values = m.Forward(window);
                    int offset = (prefix.Length - 1) * 3;
                    generated += "abc"[Enumerable.Range(0, 3).MaxBy(k => values[offset + k])];
                }
                Require(inference.Generate("ab", 8).Text == generated, "推論和訓練模式生成不一致");
                Require(inference.Generate("ab", 0).Text == "ab", "零字元生成不符");
                Console.WriteLine("推論模式、滑動視窗生成及含尾段的文本評分通過");
            }
            copy.Train(input, target);
            m.Train(input, target);
            for (int i = 0; i < m.Parameters.Count; i++)
                Require(copy.Parameters[i].Tensor.Value.GetAsArray1D().SequenceEqual(m.Parameters[i].Tensor.Value.GetAsArray1D()), "續訓不一致");
            Console.WriteLine("checkpoint 還原 logits 及下一步更新一致");
        }
        finally { if (File.Exists(file)) File.Delete(file); }

        double CpuLoss()
        {
            var result = Reference.Predict(c, 3, p, input);
            double sum = 0;
            for (int r = 0; r < input.Length; r++)
            {
                double peak = Enumerable.Range(0, 3).Max(i => result[r,i]);
                sum += Math.Log(Enumerable.Range(0, 3).Sum(i => Math.Exp(result[r,i] - peak))) + peak - result[r,target[r]];
            }
            return sum / input.Length;
        }
    }
}
