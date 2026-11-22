using ILGPU.Runtime;

namespace MiniGPT;

internal static class Checkpoint
{
    public static void Save(string path, Model model, string alphabet, string dataHash, bool overwrite = false)
    {
        string temporary = path + "." + Guid.NewGuid() + ".tmp";
        try
        {
            using (var w = new BinaryWriter(File.Create(temporary)))
            {
                w.Write("MiniGPT/1");
                var c = model.Config;
                foreach (int value in new[] { c.Batch, c.T, c.D, c.Heads, c.Layers, model.Step }) w.Write(value);
                w.Write(alphabet);
                w.Write(dataHash);
                foreach (var p in model.Parameters)
                {
                    w.Write(p.Name);
                    w.Write(p.Tensor.Value.IntExtent.X);
                    foreach (var buffer in new[] { p.Tensor.Value, p.M, p.V })
                        foreach (float f in buffer.GetAsArray1D()) w.Write(f);
                }
            }
            File.Move(temporary, path, overwrite);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    public static Model Load(string path, out string alphabet, out string dataHash)
    {
        using var r = new BinaryReader(File.OpenRead(path));
        if (r.ReadString() != "MiniGPT/1") throw new InvalidDataException("不支援的 checkpoint。");
        var c = new Config(r.ReadInt32(), r.ReadInt32(), r.ReadInt32(), r.ReadInt32(), r.ReadInt32());
        int step = r.ReadInt32();
        alphabet = r.ReadString();
        dataHash = r.ReadString();
        if (c.Batch is < 1 or > 4 || c.T is < 1 or > 64 || c.D is < 1 or > 64 ||
            c.Heads is < 1 or > 4 || c.D % c.Heads != 0 || c.Layers is < 1 or > 2 ||
            alphabet.Length is < 2 or > 1024 || alphabet.Distinct().Count() != alphabet.Length || step < 0)
            throw new InvalidDataException("模型尺寸或字表不正確。");
        var model = new Model(c, alphabet.Length) { Step = step };
        try
        {
            foreach (var p in model.Parameters)
            {
                if (r.ReadString() != p.Name || r.ReadInt32() != p.Tensor.Value.Length)
                    throw new InvalidDataException($"參數不符：{p.Name}");
                foreach (var buffer in new[] { p.Tensor.Value, p.M, p.V })
                {
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
