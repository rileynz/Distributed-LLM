#define MyAppName "Distributed LLM"
#ifndef MyAppVersion
  #define MyAppVersion "0.3.0"
#endif
#define MyAppPublisher "Distributed LLM Project"
#define MyAppExeName "DistributedLLM.exe"

[Setup]
AppId={{7CF72B25-A504-4556-92E4-A4077ED6BB73}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
VersionInfoVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\Distributed LLM
DefaultGroupName=Distributed LLM
DisableProgramGroupPage=yes
OutputDir=..\..\dist\windows
OutputBaseFilename=Distributed-LLM-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible arm64
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#MyAppExeName}
AppMutex=DistributedLLMUniversal
CloseApplications=yes
RestartApplications=no
SetupLogging=yes

[Files]
Source: "..\..\dist\windows\DistributedLLM\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Distributed LLM"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\Distributed LLM"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userstartup}\Distributed LLM"; Filename: "{app}\{#MyAppExeName}"; Parameters: "run"; Tasks: startup

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"
Name: "startup"; Description: "Start Distributed LLM when I sign in"; GroupDescription: "Startup:"; Flags: unchecked

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Distributed LLM"; Flags: nowait postinstall skipifsilent
