namespace MiniGPT;

internal static class InteractiveCli
{
    public static int Run(string[] args)
    {
        if (args.Length > 2) throw new ArgumentException("用法：chat [模型.ckpt]");
        string path = args.Length == 2 ? args[1] : DefaultCheckpoint();
        using var session = new InferenceSession(path);
        int length = 100;
        Console.WriteLine($"miniGPT | CUDA {session.Device} | step {session.Step}");
        Console.WriteLine($"模型 {Path.GetFullPath(path)}");
        Console.WriteLine("輸入字表內的文字後按 Enter 續寫。每次輸入都是獨立的新開頭，結尾空白會保留。");
        Console.WriteLine("/help 說明 | /length 60 設定新增字元數 | /info 模型資訊 | /exit 離開");
        while (true)
        {
            Console.Write("你> ");
            string? input = Console.ReadLine();
            if (input is null) break;
            if (input.Length == 0) continue;
            if (input.Trim() == "/exit") break;
            if (input.Trim() == "/help")
            {
                Console.WriteLine("每次輸入都是獨立的文字開頭。模型會逐字續寫，保留輸入結尾的空白。");
                Console.WriteLine("/length 數字：設定新增字元數，例如 /length 60。");
                Console.WriteLine("/info：顯示字表與長度；/exit：離開。");
                continue;
            }
            if (input.Trim() == "/info")
            {
                Console.WriteLine($"step {session.Step} | 新增 {length} 字元 | CUDA {session.Device}");
                Console.WriteLine($"字表 {session.Alphabet.Length} 字元：{session.Alphabet.Replace("\n", "\\n").Replace("\r", "\\r")}");
                continue;
            }
            if (input.TrimStart().StartsWith('/'))
            {
                var parts = input.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts is ["/length", var value] && int.TryParse(value, out int count) && count > 0)
                {
                    length = count;
                    Console.WriteLine($"每次新增 {length} 個字元。");
                }
                else Console.WriteLine("無效指令。使用 /help 查看說明，或 /length 60 調整長度。");
                continue;
            }
            try
            {
                Console.Write("續寫> ");
                session.Generate(input, length, c => Console.Write(c));
                Console.WriteLine();
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine(ex.Message);
                Console.WriteLine("請使用 /info 查看字表，再輸入字表內的文字。");
            }
        }
        return 0;
    }

    private static string DefaultCheckpoint()
    {
        // dotnet run 的輸出目錄位於專案內。向上找專案，讓啟動不依賴目前目錄。
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory.Parent is not null && !File.Exists(Path.Combine(directory.FullName, "MiniGPT.csproj")))
            directory = directory.Parent;
        string modelDirectory = File.Exists(Path.Combine(directory.FullName, "MiniGPT.csproj"))
            ? directory.FullName : AppContext.BaseDirectory;
        string[] files = Directory.GetFiles(modelDirectory, "*.ckpt");
        if (files.Length == 0)
            throw new ArgumentException($"在 {modelDirectory} 找不到 checkpoint。請先依第 5 課訓練模型，或在專案根目錄執行 .\\minigpt.cmd chat \"模型的完整路徑.ckpt\"。");
        if (files.Length > 1)
            throw new ArgumentException($"在 {modelDirectory} 找到 {files.Length} 個 checkpoint。請在專案根目錄執行 .\\minigpt.cmd chat \"模型路徑.ckpt\"。相對路徑以目前工作目錄為準。");
        return files[0];
    }
}
