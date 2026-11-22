namespace MiniGPT;

// 小矩陣 CPU 參考答案。用 double 與二維索引，讓 CUDA 的索引錯誤能被找到。
internal static class Reference
{
    public static double[,] Predict(Config c, int vocab, Dictionary<string, float[]> p, int[] ids)
    {
        int rows = ids.Length, d = c.D;
        var x = new double[rows, d];
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < d; j++)
                x[r,j] = (double)p["token"][ids[r] * d + j] + p["position"][r % c.T * d + j];

        for (int layer = 0; layer < c.Layers; layer++)
        {
            string prefix = $"block{layer}.";
            var n = Norm(prefix + "ln1", x);
            var q = Linear(prefix + "query", n, d);
            var k = Linear(prefix + "key", n, d);
            var v = Linear(prefix + "value", n, d);
            var mix = new double[rows, d];
            int width = d / c.Heads;
            for (int row = 0; row < rows; row++)
                for (int h = 0; h < c.Heads; h++)
                {
                    int start = row / c.T * c.T;
                    var scores = new double[row - start + 1];
                    for (int past = start; past <= row; past++)
                        for (int j = h * width; j < (h + 1) * width; j++)
                            scores[past - start] += q[row,j] * k[past,j] / Math.Sqrt(width);
                    double peak = scores.Max();
                    var exp = scores.Select(s => Math.Exp(s - peak)).ToArray();
                    double total = exp.Sum();
                    for (int past = start; past <= row; past++)
                        for (int j = h * width; j < (h + 1) * width; j++)
                            mix[row,j] += exp[past - start] / total * v[past,j];
                }
            Add(x, Linear(prefix + "project", mix, d));
            var wide = Linear(prefix + "expand", Norm(prefix + "ln2", x), 4 * d);
            for (int r = 0; r < rows; r++)
                for (int j = 0; j < 4 * d; j++)
                {
                    double a = wide[r,j];
                    wide[r,j] = 0.5 * a * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (a + 0.044715 * a * a * a)));
                }
            Add(x, Linear(prefix + "contract", wide, d));
        }
        return Linear("head", Norm("final", x), vocab);

        double[,] Linear(string name, double[,] a, int cols)
        {
            bool bias = name != "head";
            var weight = p[bias ? name + ".weight" : name];
            int inner = a.GetLength(1);
            var result = new double[rows, cols];
            for (int r = 0; r < rows; r++)
                for (int j = 0; j < cols; j++)
                {
                    result[r,j] = bias ? p[name + ".bias"][j] : 0;
                    for (int i = 0; i < inner; i++) result[r,j] += a[r,i] * weight[i * cols + j];
                }
            return result;
        }

        double[,] Norm(string name, double[,] a)
        {
            var result = new double[rows, d];
            for (int r = 0; r < rows; r++)
            {
                var values = Enumerable.Range(0, d).Select(j => a[r,j]).ToArray();
                double mean = values.Average();
                double std = Math.Sqrt(values.Average(z => (z - mean) * (z - mean)) + 1e-5);
                for (int j = 0; j < d; j++)
                    result[r,j] = (values[j] - mean) / std * p[name + ".gain"][j] + p[name + ".bias"][j];
            }
            return result;
        }

        void Add(double[,] a, double[,] b)
        {
            for (int r = 0; r < rows; r++)
                for (int j = 0; j < d; j++) a[r,j] += b[r,j];
        }
    }
}
