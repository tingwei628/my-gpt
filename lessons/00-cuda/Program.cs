using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

internal static class Program
{
    // 每次執行只負責一個位置。index 由 ILGPU 傳入。
    private static void Add(Index1D index, ArrayView<int> left,
        ArrayView<int> right, ArrayView<int> result)
    {
        result[index] = left[index] + right[index];
    }

    private static int Main()
    {
        using var context = Context.Create(builder => builder.Cuda());
        var devices = context.GetCudaDevices();
        CudaDevice? device = null;
        foreach (var candidate in devices)
        {
            Console.WriteLine($"找到 CUDA 裝置：{candidate.Name}");
            if (device is null || candidate.Name.Contains("GTX 1650"))
                device = candidate;
        }
        if (device is null)
        {
            Console.Error.WriteLine("找不到 CUDA 裝置。請執行 nvidia-smi 確認 NVIDIA 驅動程式能辨識顯卡。");
            return 1;
        }

        using var accelerator = device.CreateCudaAccelerator(context);
        Console.WriteLine($"執行裝置：{accelerator.Name} (CUDA)");
        int[] left = [1, 2];
        int[] right = [3, 4];
        using var gpuLeft = accelerator.Allocate1D(left);
        using var gpuRight = accelerator.Allocate1D(right);
        using var gpuResult = accelerator.Allocate1D<int>(left.Length);

        var add = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView<int>, ArrayView<int>, ArrayView<int>>(Add);
        add(left.Length, gpuLeft.View, gpuRight.View, gpuResult.View);
        accelerator.Synchronize();
        int[] result = gpuResult.GetAsArray1D();
        Console.WriteLine($"結果：[{string.Join(",", result)}]");
        for (int i = 0; i < left.Length; i++)
        {
            if (result[i] != left[i] + right[i])
            {
                Console.Error.WriteLine($"驗算失敗，請檢查位置 {i}。");
                return 1;
            }
        }
        return 0;
    }
}
