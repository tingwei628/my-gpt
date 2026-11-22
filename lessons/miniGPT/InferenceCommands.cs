using System.Security.Cryptography;
using System.Text.RegularExpressions;

namespace MiniGPT;

internal static class InferenceCommands
{
    public static int Run(string[] args)
    {
        if (args is ["score", var checkpoint, var textFile])
        {
            using var session = new InferenceSession(checkpoint);
            string text = File.ReadAllText(textFile).ReplaceLineEndings("\n");
            TextScore score = session.Score(text);
            Console.WriteLine($"CUDA {session.Device} | step {session.Step}");
            Console.WriteLine($"評分位置 {score.Predictions} | loss {score.Loss:F6} | 字元 perplexity {score.Perplexity:F4} | 下一字元正確率 {score.Accuracy:P2}");
            return 0;
        }
        if (args.Length < 2) throw new ArgumentException("請提供 checkpoint 路徑。");
        if (args[0] == "generate" && args.Length is 3 or 4)
        {
            int count = Count(args, 3);
            using var session = new InferenceSession(args[1]);
            Console.WriteLine(session.Generate(args[2], count).Text);
            return 0;
        }
        if (args[0] == "benchmark" && args.Length is 2 or 3)
        {
            int count = Count(args, 2);
            if (count == 0) throw new ArgumentException("benchmark 至少需生成一個字元。");
            // Windows 上持有唯讀 handle，避免續訓換檔造成權重與指紋對不上。
            using var snapshot = File.OpenRead(args[1]);
            using var session = new InferenceSession(args[1]);
            if ("the fox cat dog plays in owl ".Any(c => !session.Alphabet.Contains(c)))
                throw new ArgumentException("benchmark 的固定開頭適用於英文教材字表。自訂模型可用 generate 指定開頭，或用 score 評分。");
            Console.WriteLine($"CUDA {session.Device} | step {session.Step} | 貪婪選字 | 新增 {count} 字元");
            Console.WriteLine($"checkpoint SHA256 {Convert.ToHexString(SHA256.HashData(snapshot))}");
            Console.WriteLine($"載入 {session.LoadMs:F1} ms | 推論 buffers {session.BufferBytes / 1048576.0:F2} MiB");
            // 先執行一次，將首次啟動 GPU 的成本與暖機後生成速度分開。
            session.Generate("the ", 1);
            foreach (string prompt in new[] { "the fox ", "the cat ", "the dog plays in the ", "the owl " })
            {
                Generation generated = session.Generate(prompt, count);
                Console.WriteLine($"開頭 [{prompt}]");
                Console.WriteLine(generated.Text);
                Console.WriteLine($"首字 {generated.FirstCharacterMs:F2} ms | {generated.CharactersPerSecond:F0} 字元/s");
                int lastPeriod = generated.Text.LastIndexOf('.');
                var sentences = generated.Text[..(lastPeriod + 1)].Split('.', StringSplitOptions.RemoveEmptyEntries)
                    .Select(s => s.Trim()).ToArray();
                int valid = sentences.Count(s => Regex.IsMatch(s,
                    "^the (cat|dog|fox|owl) (sits|runs|rests|plays) in the (sun|yard|park|rain)$"));
                int repeats = sentences.Length - sentences.Distinct().Count();
                Console.WriteLine($"完整句 {sentences.Length} | 符合教材句型 {valid} | 重複句 {repeats}");
            }
            return 0;
        }
        throw new ArgumentException("用法：generate 模型.ckpt 開頭 [新增字元數] | benchmark 模型.ckpt [新增字元數] | score 模型.ckpt 文本.txt");
    }

    private static int Count(string[] args, int index)
    {
        if (args.Length == index) return 100;
        if (!int.TryParse(args[index], out int count) || count < 0)
            throw new ArgumentException("新增字元數需為大於或等於零的整數。");
        return count;
    }
}
