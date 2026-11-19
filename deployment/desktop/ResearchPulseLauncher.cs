using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

[assembly: AssemblyTitle("ResearchPulse")]
[assembly: AssemblyProduct("ResearchPulse")]
[assembly: AssemblyDescription("ResearchPulse 文献研究台启动器")]

internal static class ResearchPulseLauncher
{
    [STAThread]
    private static int Main()
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(root, "runtime", "pythonw.exe");
        string entry = Path.Combine(root, "tools", "start_desktop.pyw");
        if (!File.Exists(python) || !File.Exists(entry))
        {
            return ShowError("找不到运行组件。请完整解压 ResearchPulse，并从解压目录启动。");
        }

        try
        {
            using (Process.Start(new ProcessStartInfo
            {
                FileName = python,
                Arguments = "-IB -X utf8 \"" + entry + "\"",
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true
            }))
            {
                return 0;
            }
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return ShowError("无法启动研究台。请检查解压目录的访问权限和运行组件是否完整。");
        }
    }

    private static int ShowError(string message)
    {
        MessageBox.Show(message, "ResearchPulse", MessageBoxButtons.OK, MessageBoxIcon.Error);
        return 1;
    }
}
