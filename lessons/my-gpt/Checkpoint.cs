using ILGPU.Runtime;

namespace MyGpt;

internal static class Checkpoint
{
    public static void Save(string path, Model model, string alphabet, string dataHash, bool overwrite = false)
    {
        if (!model.ForTraining) throw new InvalidOperationException("儲存續訓 checkpoint 需要訓練模式。");
        string temporary = path + "." + Guid.NewGuid() + ".tmp";
        try
        {
            using (var w = new BinaryWriter(File.Create(temporary)))
            {
                w.Write("my-gpt/1");
                var c = model.Config;
                foreach (int value in new[] { c.Batch, c.T, c.D, c.Heads, c.Layers, model.Step }) w.Write(value);
                w.Write(alphabet);
                w.Write(dataHash);
                foreach (var p in model.Parameters)
                {
                    w.Write(p.Name);
                    w.Write(p.Tensor.Value.IntExtent.X);
                    foreach (var buffer in new[] { p.Tensor.Value, p.M!, p.V! })
                        foreach (float f in buffer.GetAsArray1D()) w.Write(f);
                }
            }
            File.Move(temporary, path, overwrite);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    public static Model Load(string path, out string alphabet, out string dataHash, bool forTraining = true)
    {
        using var r = new BinaryReader(File.OpenRead(path));
        string format = r.ReadString();
        // 相容更名前產生的權重，讓既有訓練成果能繼續使用。
        if (format is not ("my-gpt/1" or "MiniGPT/1")) throw new InvalidDataException("不支援的 checkpoint。");
        var c = new Config(r.ReadInt32(), r.ReadInt32(), r.ReadInt32(), r.ReadInt32(), r.ReadInt32());
        int step = r.ReadInt32();
        alphabet = r.ReadString();
        dataHash = r.ReadString();
        if (c.Batch is < 1 or > 4 || c.T is < 1 or > 64 || c.D is < 1 or > 64 ||
            c.Heads is < 1 or > 4 || c.D % c.Heads != 0 || c.Layers is < 1 or > 2 ||
            alphabet.Length is < 2 or > 1024 || alphabet.Distinct().Count() != alphabet.Length || step < 0)
            throw new InvalidDataException("模型尺寸或字表不正確。");
        var model = new Model(forTraining ? c : c with { Batch = 1 }, alphabet.Length, forTraining) { Step = step };
        try
        {
            foreach (var p in model.Parameters)
            {
                if (r.ReadString() != p.Name || r.ReadInt32() != p.Tensor.Value.Length)
                    throw new InvalidDataException($"參數不符：{p.Name}");
                foreach (var buffer in new[] { p.Tensor.Value, p.M, p.V })
                {
                    if (buffer is null)
                    {
                        // 舊 checkpoint 的動量仍佔據檔案空間，推論時直接跳過。
                        long bytes = p.Tensor.Value.Length * sizeof(float);
                        if (bytes > r.BaseStream.Length - r.BaseStream.Position) throw new EndOfStreamException("checkpoint 資料不完整。");
                        r.BaseStream.Seek(bytes, SeekOrigin.Current);
                        continue;
                    }
                    var values = new float[buffer.Length];
                    for (int i = 0; i < values.Length; i++) values[i] = r.ReadSingle();
                    buffer.CopyFromCPU(values);
                }
            }
            if (r.BaseStream.Position != r.BaseStream.Length) throw new InvalidDataException("多餘的模型資料。");
            return model;
        }
        catch { model.Dispose(); throw; }
    }
}
