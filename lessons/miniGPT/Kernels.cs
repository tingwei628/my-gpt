using ILGPU;
using ILGPU.Algorithms;

namespace MiniGPT;

// A matrix has Rows rows. Linear maps K input columns to N output columns.
// For attention, rows are contiguous sequences of T tokens with H heads of width K.
public enum Op { Linear, Add, Gain, Bias, Embed, Norm, Gelu, Scores, Softmax, Mix, Loss }
public readonly record struct Shape(int Rows, int K, int N, int T = 1, int H = 1);
public readonly record struct Values(ArrayView<float> A, ArrayView<float> B, ArrayView<int> Ids);

public static class Kernels
{
    private static float TanhTerm(float x) => XMath.Tanh(0.79788456f * (x + 0.044715f * x * x * x));

    private static void Statistics(ArrayView<float> x, int start, int width, out float mean, out float inv)
    {
        mean = 0;
        for (int j = 0; j < width; j++) mean += x[start + j];
        mean /= width;
        float variance = 0;
        for (int j = 0; j < width; j++)
        {
            float centered = x[start + j] - mean;
            variance += centered * centered;
        }
        inv = 1 / XMath.Sqrt(variance / width + 1e-5f);
    }

    // One worker owns one output, so no two threads write the same element.
    public static void Forward(Index1D index, Op op, Shape s, Values v, ArrayView<float> output)
    {
        int i = index, row = i / s.N, col = i % s.N;
        float sum = 0;
        switch (op)
        {
            case Op.Linear:
                for (int j = 0; j < s.K; j++) sum += v.A[row * s.K + j] * v.B[j * s.N + col];
                break;
            case Op.Add: sum = v.A[i] + v.B[i]; break;
            case Op.Gain: sum = v.A[i] * v.B[col]; break;
            case Op.Bias: sum = v.A[i] + v.B[col]; break;
            case Op.Embed: sum = v.A[v.Ids[row] * s.N + col]; break;
            case Op.Gelu: sum = 0.5f * v.A[i] * (1 + TanhTerm(v.A[i])); break;
            case Op.Norm:
                Statistics(v.A, row * s.N, s.N, out float mean, out float inv);
                sum = (v.A[i] - mean) * inv;
                break;
            case Op.Scores:
                // Scores layout: [batch, query position, head, key position].
                int query = i / (s.H * s.T), head = (i / s.T) % s.H, key = i % s.T;
                int keyRow = query / s.T * s.T + key;
                for (int j = 0; j < s.K; j++)
                    sum += v.A[(query * s.H + head) * s.K + j] * v.B[(keyRow * s.H + head) * s.K + j];
                sum /= XMath.Sqrt((float)s.K);
                break;
            case Op.Softmax:
                int start = i - col;
                int allowed = row / s.H % s.T + 1;
                if (col < allowed)
                {
                    float max = v.A[start];
                    for (int j = 1; j < allowed; j++) max = XMath.Max(max, v.A[start + j]);
                    float total = 0;
                    for (int j = 0; j < allowed; j++) total += XMath.Exp(v.A[start + j] - max);
                    sum = XMath.Exp(v.A[i] - max) / total;
                }
                break;
            case Op.Mix:
                int h = col / s.K;
                for (int j = 0; j <= row % s.T; j++)
                    sum += v.A[(row * s.H + h) * s.T + j] * v.B[(row / s.T * s.T + j) * s.N + col];
                break;
            case Op.Loss:
                int offset = i * s.K;
                float peak = v.A[offset];
                for (int j = 1; j < s.K; j++) peak = XMath.Max(peak, v.A[offset + j]);
                for (int j = 0; j < s.K; j++) sum += XMath.Exp(v.A[offset + j] - peak);
                sum = (peak - v.A[offset + v.Ids[i]]) + XMath.Log(sum);
                break;
        }
        output[i] = sum;
    }

    // Each worker collects every contribution to ONE input gradient.
    // side=0 differentiates A, side=1 differentiates B. The graph may reuse an input.
    public static void Backward(Index1D index, Op op, Shape s, Values v,
        ArrayView<float> y, ArrayView<float> dy, int side, ArrayView<float> dx)
    {
        int i = index;
        float sum = 0;
        switch (op)
        {
            case Op.Linear:
                if (side == 0)
                    for (int j = 0; j < s.N; j++) sum += dy[i / s.K * s.N + j] * v.B[i % s.K * s.N + j];
                else
                    for (int r = 0; r < s.Rows; r++) sum += v.A[r * s.K + i / s.N] * dy[r * s.N + i % s.N];
                break;
            case Op.Add: sum = dy[i]; break;
            case Op.Bias:
            case Op.Gain:
                if (side == 0) sum = dy[i] * (op == Op.Gain ? v.B[i % s.N] : 1);
                else
                    for (int j = i; j < s.Rows * s.N; j += s.N)
                        sum += dy[j] * (op == Op.Gain ? v.A[j] : 1);
                break;
            case Op.Embed:
                for (int r = 0; r < s.Rows; r++)
                    if (v.Ids[r] == i / s.N) sum += dy[r * s.N + i % s.N];
                break;
            case Op.Gelu:
                float x = v.A[i], u = TanhTerm(x);
                sum = dy[i] * (0.5f * (1 + u) + 0.5f * x * (1 - u * u) * 0.79788456f * (1 + 0.134145f * x * x));
                break;
            case Op.Norm:
                int start = i / s.N * s.N;
                Statistics(v.A, start, s.N, out _, out float inv);
                float g = 0, gy = 0;
                for (int j = start; j < start + s.N; j++) { g += dy[j]; gy += dy[j] * y[j]; }
                sum = inv * (dy[i] - (g + y[i] * gy) / s.N);
                break;
            case Op.Scores:
                int token = i / (s.H * s.K), head = i / s.K % s.H, dim = i % s.K;
                int first = token / s.T * s.T;
                for (int j = 0; j < s.T; j++)
                {
                    int query = side == 0 ? token : first + j;
                    int key = side == 0 ? first + j : token;
                    float other = side == 0 ? v.B[(key * s.H + head) * s.K + dim] : v.A[(query * s.H + head) * s.K + dim];
                    sum += dy[(query * s.H + head) * s.T + key % s.T] * other;
                }
                sum /= XMath.Sqrt((float)s.K);
                break;
            case Op.Softmax:
                int rowStart = i / s.N * s.N;
                for (int j = rowStart; j < rowStart + s.N; j++) sum += dy[j] * y[j];
                sum = y[i] * (dy[i] - sum);
                break;
            case Op.Mix:
                if (side == 0)
                {
                    int query = i / (s.H * s.T), h = i / s.T % s.H, key = i % s.T;
                    if (key <= query % s.T)
                        for (int j = h * s.K; j < (h + 1) * s.K; j++)
                            sum += dy[query * s.N + j] * v.B[(query / s.T * s.T + key) * s.N + j];
                }
                else
                {
                    int key = i / s.N, col = i % s.N;
                    for (int j = key % s.T; j < s.T; j++)
                    {
                        int query = key / s.T * s.T + j;
                        sum += dy[query * s.N + col] * v.A[(query * s.H + col / s.K) * s.T + key % s.T];
                    }
                }
                break;
            case Op.Loss:
                int offset = i / s.K * s.K;
                float peak = v.A[offset], total = 0;
                for (int j = 1; j < s.K; j++) peak = XMath.Max(peak, v.A[offset + j]);
                for (int j = 0; j < s.K; j++) total += XMath.Exp(v.A[offset + j] - peak);
                sum = dy[i / s.K] * (XMath.Exp(v.A[i] - peak) / total - (i % s.K == v.Ids[i / s.K] ? 1 : 0));
                break;
        }
        dx[i] += sum;
    }

    public static void Adam(Index1D i, ArrayView<float> w, ArrayView<float> grad,
        ArrayView<float> m, ArrayView<float> v, float correction1, float correction2, float decay)
    {
        float g = grad[i];
        m[i] = 0.9f * m[i] + 0.1f * g;
        v[i] = 0.999f * v[i] + 0.001f * g * g;
        w[i] -= 0.003f * (m[i] / correction1 / (XMath.Sqrt(v[i] / correction2) + 1e-8f) + decay * w[i]);
    }
}
