param(
    [ValidateSet('Prompt', 'Enable', 'Disable', 'Preview')]
    [string]$Action = 'Prompt'
)

$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$sid = $identity.User.Value
$taskName = "ResearchPulse Desktop Daily ($sid)"
if ($Action -eq 'Prompt') {
    Write-Host '每日检查安排在北京时间 05:00，使用当前 Windows 用户身份。'
    Write-Host '输入 1 启用，输入 2 移除定时任务，按回车退出。'
    switch (Read-Host '请选择') {
        '1' { $Action = 'Enable' }
        '2' { $Action = 'Disable' }
        default { exit 0 }
    }
}
if ($Action -eq 'Disable') {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    }
    Write-Host '已移除 ResearchPulse 定时任务。'
    exit 0
}

$root = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $root 'runtime\pythonw.exe'
$entryPath = Join-Path $PSScriptRoot 'start_background.pyw'
if (!(Test-Path -LiteralPath $pythonPath -PathType Leaf) -or !(Test-Path -LiteralPath $entryPath -PathType Leaf)) {
    throw '找不到研究台运行环境，请先完整解压安装包。'
}
$command = [Security.SecurityElement]::Escape($pythonPath)
$arguments = [Security.SecurityElement]::Escape("-IB -X utf8 `"$entryPath`"")
$workingDir = [Security.SecurityElement]::Escape($root)
$timezone = [TimeZoneInfo]::FindSystemTimeZoneById('China Standard Time')
$now = [TimeZoneInfo]::ConvertTime([DateTimeOffset]::Now, $timezone)
$first = [DateTimeOffset]::new($now.Year, $now.Month, $now.Day, 5, 0, 0, $now.Offset)
if ($first -le $now) {
    $first = $first.AddDays(1)
}
$boundary = $first.ToString('yyyy-MM-ddTHH:mm:sszzz', [Globalization.CultureInfo]::InvariantCulture)
$xml = @"
<Task version="1.3" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>ResearchPulse 北京时间 05:00 日报检查</Description></RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>$boundary</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>$sid</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
  </Settings>
  <Actions Context="Author">
    <Exec><Command>$command</Command><Arguments>$arguments</Arguments><WorkingDirectory>$workingDir</WorkingDirectory></Exec>
  </Actions>
</Task>
"@
if ($Action -eq 'Preview') {
    $xml
    exit 0
}
Register-ScheduledTask -TaskName $taskName -Xml $xml -Force | Out-Null
Write-Host "已启用每日检查，首次计划时间 $boundary。项目仍需各自开启每日检测。"
