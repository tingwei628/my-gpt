using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Buffer = ILGPU.Runtime.MemoryBuffer1D<float, ILGPU.Stride1D.Dense>;
using IdBuffer = ILGPU.Runtime.MemoryBuffer1D<int, ILGPU.Stride1D.Dense>;

namespace MiniGPT;

internal record Config(int Batch = 1, int T = 64, int D = 64, int Heads = 4, int Layers = 2);
internal sealed record Tensor(Buffer Value, Buffer? Grad);
internal sealed record Parameter(string Name, Tensor Tensor, Buffer? M, Buffer? V, float Decay);
internal sealed record Node(Op Op, Shape Shape, Tensor A, Tensor? B, IdBuffer Ids, Tensor Output);

internal sealed class Model : IDisposable
{
    private readonly Context context;
    private readonly Accelerator gpu;
    private readonly Random random = new(42);
    private readonly List<IDisposable> buffers = [];
    private readonly List<Node> graph = [];
    public List<Parameter> Parameters { get; } = [];
    public Config Config { get; }
    public bool ForTraining { get; }
    public int Step { get; set; }
    public Tensor Logits { get; }
    public Tensor Loss { get; }
    public string Device => gpu.Name;
    public long BufferBytes { get; private set; }
    private readonly IdBuffer tokens, targets, positions;
    private readonly Action<Index1D, Op, Shape, Values, ArrayView<float>> forward;
    private readonly Action<Index1D, Op, Shape, Values, ArrayView<float>, ArrayView<float>, int, ArrayView<float>>? backward;
    public const float DefaultLearningRate = 0.003f;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float, float, float>? adam;

    public Model(Config c, int vocab, bool forTraining = true)
    {
        Config = c;
        ForTraining = forTraining;
        context = Context.Create(b => b.Cuda().EnableAlgorithms());
        CudaDevice? device = null;
        foreach (var candidate in context.GetCudaDevices())
            if (device is null || candidate.Name.Contains("GTX 1650")) device = candidate;
        gpu = (device ?? throw new InvalidOperationException("找不到 CUDA 裝置，請先完成第 0 課。")).CreateCudaAccelerator(context);
        forward = gpu.LoadAutoGroupedStreamKernel<Index1D, Op, Shape, Values, ArrayView<float>>(Kernels.Forward);
        if (forTraining)
        {
            backward = gpu.LoadAutoGroupedStreamKernel<Index1D, Op, Shape, Values, ArrayView<float>, ArrayView<float>, int, ArrayView<float>>(Kernels.Backward);
            adam = gpu.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float, float, float>(Kernels.Adam);
        }

        int rows = c.Batch * c.T, d = c.D;
        tokens = Ids(new int[rows]);
        targets = Ids(new int[rows]);
        positions = Ids(Enumerable.Range(0, rows).Select(i => i % c.T).ToArray());
        var word = Apply(Op.Embed, new(rows, vocab, d), Param("token", vocab * d), ids: tokens);
        var pos = Apply(Op.Embed, new(rows, c.T, d), Param("position", c.T * d), ids: positions);
        var x = Apply(Op.Add, new(rows, d, d), word, pos);
        for (int block = 0; block < c.Layers; block++)
        {
            string name = $"block{block}.";
            var n = Normalize(name + "ln1", x);
            var q = Dense(name + "query", n, d, d);
            var k = Dense(name + "key", n, d, d);
            var v = Dense(name + "value", n, d, d);
            var scores = Apply(Op.Scores, new(rows * c.Heads, d / c.Heads, c.T, c.T, c.Heads), q, k);
            var weights = Apply(Op.Softmax, new(rows * c.Heads, c.T, c.T, c.T, c.Heads), scores);
            var mix = Apply(Op.Mix, new(rows, d / c.Heads, d, c.T, c.Heads), weights, v);
            x = Apply(Op.Add, new(rows, d, d), x, Dense(name + "project", mix, d, d));
            var wide = Dense(name + "expand", Normalize(name + "ln2", x), d, 4 * d);
            var activated = Apply(Op.Gelu, new(rows, 4 * d, 4 * d), wide);
            x = Apply(Op.Add, new(rows, d, d), x, Dense(name + "contract", activated, 4 * d, d));
        }
        Logits = Apply(Op.Linear, new(rows, d, vocab), Normalize("final", x), Param("head", d * vocab));
        Loss = Apply(Op.Loss, new(rows, vocab, 1), Logits, ids: targets);
        Loss.Grad?.CopyFromCPU(Enumerable.Repeat(1f / rows, rows).ToArray());
    }

    private Buffer Allocate(int count)
    {
        var buf = gpu.Allocate1D<float>(count);
        buffers.Add(buf);
        BufferBytes += 4L * count;
        return buf;
    }

    private IdBuffer Ids(int[] values)
    {
        var buf = gpu.Allocate1D(values);
        buffers.Add(buf);
        BufferBytes += 4L * values.Length;
        return buf;
    }

    private Tensor TensorOf(int count) => new(Allocate(count), ForTraining ? Allocate(count) : null);

    private Tensor Param(string name, int count, float? fill = null)
    {
        var values = TensorOf(count);
        values.Value.CopyFromCPU(Enumerable.Range(0, count).Select(i => fill ?? (random.NextSingle() - 0.5f) * 0.07f).ToArray());
        var m = ForTraining ? Allocate(count) : null;
        var v = ForTraining ? Allocate(count) : null;
        m?.MemSetToZero();
        v?.MemSetToZero();
        Parameters.Add(new(name, values, m, v, fill.HasValue ? 0 : 0.01f));
        return values;
    }

    private Tensor Apply(Op op, Shape shape, Tensor a, Tensor? b = null, IdBuffer? ids = null)
    {
        var output = TensorOf(shape.Rows * shape.N);
        graph.Add(new(op, shape, a, b, ids ?? tokens, output));
        return output;
    }

    private Tensor Dense(string name, Tensor x, int input, int output)
    {
        var shape = new Shape(Config.T * Config.Batch, input, output);
        var product = Apply(Op.Linear, shape, x, Param(name + ".weight", input * output));
        return Apply(Op.Bias, shape, product, Param(name + ".bias", output, 0));
    }

    private Tensor Normalize(string name, Tensor x)
    {
        int d = Config.D;
        var shape = new Shape(Config.T * Config.Batch, d, d);
        var centered = Apply(Op.Norm, shape, x);
        var scaled = Apply(Op.Gain, shape, centered, Param(name + ".gain", d, 1));
        return Apply(Op.Bias, shape, scaled, Param(name + ".bias", d, 0));
    }

    private static Values Inputs(Node n) => new(n.A.Value.View, (n.B ?? n.A).Value.View, n.Ids.View);

    public float Evaluate(int[] input, int[] labels)
    {
        if (input.Length != tokens.Length || labels.Length != targets.Length)
            throw new ArgumentException("輸入與目標長度應等於 Batch × T。");
        tokens.CopyFromCPU(input);
        targets.CopyFromCPU(labels);
        foreach (var n in graph)
            forward(n.Output.Value.IntExtent, n.Op, n.Shape, Inputs(n), n.Output.Value.View);
        return Loss.Value.GetAsArray1D().Average();
    }

    // 推論只跑到 logits，省去標籤上傳與交叉熵計算。
    public float[] Forward(int[] input)
    {
        if (input.Length != tokens.Length)
            throw new ArgumentException("輸入長度應等於 Batch × T。");
        tokens.CopyFromCPU(input);
        foreach (var n in graph)
        {
            if (n.Op == Op.Loss) break;
            forward(n.Output.Value.IntExtent, n.Op, n.Shape, Inputs(n), n.Output.Value.View);
        }
        return Logits.Value.GetAsArray1D();
    }

    public void Differentiate()
    {
        if (!ForTraining) throw new InvalidOperationException("反向傳播需要訓練模式。");
        foreach (var p in Parameters) p.Tensor.Grad!.MemSetToZero();
        foreach (var n in graph)
            if (n.Output != Loss) n.Output.Grad!.MemSetToZero();
        for (int i = graph.Count - 1; i >= 0; i--)
        {
            var n = graph[i];
            Pass(n.A, 0);
            if (n.B is not null) Pass(n.B, 1);
            void Pass(Tensor input, int side) =>
                backward!(input.Grad!.IntExtent, n.Op, n.Shape, Inputs(n), n.Output.Value.View,
                    n.Output.Grad!.View, side, input.Grad.View);
        }
    }

    public float Train(int[] input, int[] labels, float learningRate = DefaultLearningRate)
    {
        float loss = Evaluate(input, labels);
        if (!float.IsFinite(loss)) throw new ArithmeticException("loss 超出浮點數範圍。");
        Differentiate();
        Step++;
        foreach (var p in Parameters)
            adam!(p.Tensor.Value.IntExtent, p.Tensor.Value.View, p.Tensor.Grad!.View, p.M!.View, p.V!.View,
                1 - MathF.Pow(0.9f, Step), 1 - MathF.Pow(0.999f, Step), p.Decay, learningRate);
        return loss;
    }

    public void Sync() => gpu.Synchronize();

    public void ShowBlock()
    {
        foreach (var n in graph)
        {
            Console.WriteLine($"{n.Op,-8} {n.Shape.Rows} × {n.Shape.N}");
            if (n.Op != Op.Softmax) continue;
            var weights = n.Output.Value.GetAsArray1D();
            Console.WriteLine("第一組文字、第 1 個頭，每一列對應一個預測位置：");
            for (int r = 0; r < Config.T; r++)
            {
                int start = r * Config.Heads * Config.T;
                Console.WriteLine(string.Join(" ", weights[start..(start + Config.T)].Select(x => x.ToString("F3"))));
            }
        }
    }

    public void ShowLinear()
    {
        using var a = gpu.Allocate1D(new float[] { 1, 2, 3, 4 });
        using var b = gpu.Allocate1D(new float[] { 2, 0, 1, 3 });
        using var y = gpu.Allocate1D<float>(4);
        forward(4, Op.Linear, new(2, 2, 2), new(a.View, b.View, tokens.View), y.View);
        var result = y.GetAsArray1D();
        Console.WriteLine($"GPU 矩陣乘積：[{string.Join(", ", result)}]");
        if (!result.SequenceEqual(new float[] { 4, 6, 10, 12 }))
            throw new InvalidOperationException("矩陣乘法與手算不符。");
    }

    public void Dispose()
    {
        foreach (var buffer in buffers) buffer.Dispose();
        gpu.Dispose();
        context.Dispose();
    }
}
