#Requires -RunAsAdministrator
param([ValidateRange(1024,65535)][int]$Port = 8765)

$ErrorActionPreference = 'Stop'
$nodePath = (Get-Command node.exe -CommandType Application -ErrorAction Stop).Source
$ruleName = "PokeRL-HumanPlay-TCP-$Port"
$ruleOptions = @{
    Direction = 'Inbound'
    Action = 'Allow'
    Enabled = 'True'
    Protocol = 'TCP'
    LocalPort = $Port
    Program = $nodePath
    Profile = 'Private'
    RemoteAddress = 'LocalSubnet'
}
if (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue) {
    Set-NetFirewallRule -Name $ruleName @ruleOptions
} else {
    New-NetFirewallRule -Name $ruleName -DisplayName "Poke-RL human play (TCP $Port, LAN)" @ruleOptions | Out-Null
}
Write-Host "Allowed Poke-RL on TCP $Port from the local subnet on Private networks."
