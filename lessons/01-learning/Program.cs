// y = w * x。目標是 x=2 時輸出 4，loss 是誤差平方的一半。
double weight = 1;
double prediction = weight * 2;
double error = prediction - 4;
double gradient = error * 2;
weight -= 0.1 * gradient;
Console.WriteLine($"權重 1 → {weight:F1}，loss {error * error / 2:F2} → {Math.Pow(weight * 2 - 4, 2) / 2:F2}");

// 每列代表目前的字，每欄代表下一字。分數越高，預測機率越大。
const string alphabet = "abc", sample = "abcab";
var scores = new double[3, 3];
for (int step = 0; step < 100; step++)
{
    for (int pos = 0; pos < sample.Length - 1; pos++)
    {
        int current = alphabet.IndexOf(sample[pos]);
        int target = alphabet.IndexOf(sample[pos + 1]);
        double max = Enumerable.Range(0, 3).Max(i => scores[current, i]);
        var exp = Enumerable.Range(0, 3).Select(i => Math.Exp(scores[current, i] - max)).ToArray();
        double sum = exp.Sum();
        for (int i = 0; i < 3; i++)
        {
            double probability = exp[i] / sum;
            scores[current, i] -= 0.1 * (probability - (i == target ? 1 : 0));
        }
    }
}
foreach (char c in alphabet)
{
    int next = Enumerable.Range(0, 3).MaxBy(i => scores[alphabet.IndexOf(c), i]);
    Console.WriteLine($"{c} → {alphabet[next]}");
}
